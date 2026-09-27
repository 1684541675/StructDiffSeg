import os, cv2, torch, numpy as np,random
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from torch import nn
from vit_seg_configs import get_r50_b16_config
from unet import network
import torch.nn.functional as F
from process import ISIC,ACDC2DDataset,ACDC2DDataset1
from torch.amp import autocast, GradScaler
from  calpixel import calculate_class_weights
from sklearn.metrics import confusion_matrix
# ---------- 3. 训练 ----------


class DiceCELoss1(nn.Module):
    def __init__(self, n_classes=2, dice_weight=0.5, ce_weight=0.5, smooth=1e-5, lesion_class_idx=1, class_balance_weights=None, device='cuda'):
        super().__init__()
        # ... (初始化部分保持不变) ...
        self.n_classes = n_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.lesion_class_idx = lesion_class_idx  # 病灶类索引（默认1，二分类场景）
        if class_balance_weights is None and n_classes == 2:
            self.class_balance_weights = torch.tensor([0.1, 0.9], dtype=torch.float32).to(device)
        else:
            self.class_balance_weights = class_balance_weights.to(device)
        
        # 关键修改：reduction='none'，为了保留像素级 Loss
        self.ce = nn.CrossEntropyLoss(weight=self.class_balance_weights, reduction='none')

    def forward(self, logits, target, uncertainty=None, scale=0.0):
        """
        logits: (B, 2, H, W)
        target: (B, 1, H, W)
        uncertainty: (B, H, W)
        scale: float
        """
        # 1. CE Loss (Pixel-wise)
        target_ce = target.squeeze(1).long()  # (B, H, W)
        loss_ce_map = self.ce(logits, target_ce)  # 结果也是 (B, H, W)
        #print(loss_ce_map.shape)
        # 2. 不确定性加权 (只针对 CE Loss)
        if uncertainty is not None and scale > 0:
            # 维度匹配：loss_ce_map [B,H,W] * weight [B,H,W]
            # 论文公式：Weight = 1 + eta * U
            weight_map = 1.0 + scale * uncertainty
            loss_ce = (loss_ce_map * weight_map).mean()
        else:
            loss_ce = loss_ce_map.mean()

        # 3. Dice Loss (保持不变，Dice通常无法像素级加权)
        pred = torch.softmax(logits, dim=1)
        pred_lesion = pred[:, self.lesion_class_idx, :, :]
        target_lesion = (target_ce == self.lesion_class_idx).float()
        
        intersection = (pred_lesion * target_lesion).sum(dim=(1, 2))
        card_sum = pred_lesion.sum(dim=(1, 2)) + target_lesion.sum(dim=(1, 2))
        dice_lesion = (2. * intersection + self.smooth) / (card_sum + self.smooth)
        loss_dice = 1 - dice_lesion.mean()

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


class MultiClassDiceCELoss(nn.Module):
    def __init__(self, n_classes=4, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # inputs: [B, 4, H, W] (Logits)
        # targets: [B, H, W] (Long)
        
        # 1. Cross Entropy Loss
        ce = self.ce_loss(inputs, targets)
        
        # 2. Dice Loss
        # 先做 Softmax 拿到概率
        probs = F.softmax(inputs, dim=1)
        
        # 将 target 转为 One-hot 编码: [B, H, W] -> [B, 4, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        
        # 计算每个类别的 Dice
        # intersection: [B, 4]
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        smooth = 1e-5
        dice_score = (2. * intersection + smooth) / (union + smooth)
        
        # 对所有类别取平均 (也可以只取后3类前景)
        dice_mean = dice_score.mean()
        
        loss = self.ce_weight * ce + self.dice_weight * (1 - dice_mean)
        return loss

@torch.no_grad()
def metrics_logits(logits, target, smooth=1e-7, lesion_class_idx=1):
    """
    适配医学图像二分类分割的指标计算函数（聚焦病灶类，避免背景类干扰）
    核心：仅计算病灶类（默认索引1）的指标，贴合临床需求（如皮肤病变、肿瘤分割等）
    
    Parameters:
        logits: (B, C, H, W)  模型输出的logits（二分类时C=2，通道0=背景，通道1=病灶）
        target: (B, 1, H, W)  真实标签mask（值为0/1，0=背景，1=病灶）
        smooth: float  平滑项，避免除0错误
        lesion_class_idx: int  病灶类的通道索引（二分类默认1，若标签定义相反可改为0）
    
    Returns:
        dice: float  病灶类Dice系数（核心临床指标，0~1，越接近1越好）
        iou: float   病灶类IoU（Jaccard系数，0~1）
        recall: float  病灶类灵敏度（Sen，防漏检，0~1）
        precision: float  病灶类精确率（Spe无直接返回，需单独计算，0~1）
    """
    # 1. 模型输出转概率+预测标签（规范流程，确保预测可靠）
    prob = torch.softmax(logits, dim=1)  # (B,C,H,W) → 通道维转概率（0~1）
    pred = prob.argmax(dim=1)            # (B,H,W) → 取概率最大类作为预测标签（0=背景，1=病灶）

    # 2. 标签格式适配（压缩维度+转长整数，避免维度错误）
    target = target.squeeze(1).long()  # (B,1,H,W) → (B,H,W)，转long适配one-hot编码

    # 3. 仅统计【病灶类】的TP/FP/FN（关键：忽略背景类，聚焦临床核心）
    # 病灶类真实标签：target == lesion_class_idx → (B,H,W)（bool）
    # 病灶类预测标签：pred == lesion_class_idx → (B,H,W)（bool）
    tp = (pred == lesion_class_idx) & (target == lesion_class_idx)  # 真阳性（病灶预测对）
    fp = (pred == lesion_class_idx) & (target != lesion_class_idx)  # 假阳性（背景误判为病灶）
    fn = (pred != lesion_class_idx) & (target == lesion_class_idx)  # 假阴性（病灶漏判为背景）

    # 4. 空间维度求和（每个样本的病灶类总TP/FP/FN）→ (B,)
    tp_sum = tp.float().sum(dim=(1, 2))  # 每个样本的病灶类TP总数
    fp_sum = fp.float().sum(dim=(1, 2))  # 每个样本的病灶类FP总数
    fn_sum = fn.float().sum(dim=(1, 2))  # 每个样本的病灶类FN总数

    # 5. 计算病灶类指标（batch内样本平均，反映整体临床性能）
    # Dice系数（核心指标，平衡漏检和误检）
    dice = ((2 * tp_sum + smooth) / (2 * tp_sum + fp_sum + fn_sum + smooth)).mean()
    # IoU（交并比，反映分割区域重合度）
    iou = ((tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)).mean()

    # 6. 转成numpy scalar返回（方便后续统计）
    return dice.item(), iou.item()

@torch.no_grad()
def calculate_metrics(logits, targets, n_classes=4):
    """计算每个类别的 Dice"""
    probs = F.softmax(logits, dim=1) # [B, 4, H, W]
    preds = torch.argmax(probs, dim=1) # [B, H, W]
    
    # 转换 targets 为 one-hot 用于计算 Dice
    targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2)
    preds_one_hot = F.one_hot(preds, num_classes=n_classes).permute(0, 3, 1, 2)
    
    dices = []
    # 忽略背景(0)，只看 1, 2, 3
    for i in range(1, n_classes):
        pred_i = preds_one_hot[:, i, :, :]
        target_i = targets_one_hot[:, i, :, :]
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        smooth = 1e-5
        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice.item())
        
    return dices # 返回 [Dice_RV, Dice_Myo, Dice_LV]



class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.005, restore_best=True):
        self.patience = patience  # 容忍轮数
        self.min_delta = min_delta  # 最小提升阈值
        self.restore_best = restore_best
        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience} - 指标未提升')
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best:
                    print('恢复最佳模型参数...')
                    model.load_state_dict(self.best_state)
        else:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
            
    def _save_checkpoint(self, model):
        self.best_state = model.state_dict().copy()

def set_seed(seed=42):
    """固定所有随机种子以确保实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = EarlyStopping()
    config_vit = get_r50_b16_config()
    img_size=  256
    # 2. 剩余字段硬编码
    config_vit.n_classes = 4                 # ISIC 二分类
    config_vit.n_skip = 3
    config_vit.batch_size = 4                # 按 GPU 调
    # 3. 补丁尺寸写死（H=256, W=256, patch=16）
    patch_size = 16
    h, w = img_size,img_size
    config_vit.patches.grid = (h // patch_size, w // patch_size)
    config_vit.n_patches = (h // patch_size) * (w // patch_size)
    config_vit.h = h // patch_size
    config_vit.w = w // patch_size
    batch = 2
    epochs = 1
    lr = 1e-3
    early_stopping = EarlyStopping()
    num_workers = 4  # 自动适配CPU核心数
    
    DATA_ROOT = "../ACDC2D"  # 请确认您的路径
    train_ds = ACDC2DDataset1(DATA_ROOT, split='train', img_size=img_size)
    test_ds = ACDC2DDataset1(DATA_ROOT, split='val', img_size=img_size)
    train_ds.files=train_ds.files[:50]
    test_ds.files=test_ds.files[:50]
    train_loader = DataLoader(train_ds, batch, shuffle=True, 
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=2  
                            )
    val_loader   = DataLoader(test_ds,   batch, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)
    model = network(1,4,config=config_vit).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # 学习率调度器：CosineAnnealingLR（T_max=50，min_lr=1e-5）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, T_max=50, eta_min=1e-5
    )
    
    #loss_fn = DiceCELoss1(device=device,class_balance_weights=class_balance_weights)
    loss_fn = MultiClassDiceCELoss(n_classes=4).to(device)
    
   
    best = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(epochs):
        if early_stopping.should_stop:
            print(f'早停触发！在第{epoch+1}轮停止训练')
            break
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            #with torch.amp.autocast(device_type="cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

       
       

        # 验证
        model.eval()
       
        dice_scores = [] # 存每张图的 [RV, Myo, LV] 得分
        

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Val')
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                #with torch.amp.autocast(device_type="cuda"):
                pred = model(x)                            # (B,2,H,W)
               
                batch_dices = calculate_metrics(pred, y, n_classes=4)
                dice_scores.append(batch_dices)
                
                # 可选：在进度条上实时显示当前批次的Dice（更直观）
                val_pbar.set_postfix()
        
        dice_scores = np.array(dice_scores) # [Batch数, 3]
        avg_dices = np.mean(dice_scores, axis=0) # [RV平均, Myo平均, LV平均]
        dice_mean = np.mean(avg_dices)           # 最终平均分
        
        print(f"\n[Val] | Mean Dice: {dice_mean:.4f}")
        print(f"      RV: {avg_dices[0]:.4f} | Myo: {avg_dices[1]:.4f} | LV: {avg_dices[2]:.4f}")
        
        # CosineAnnealingLR每个epoch更新一次学习率（必须在验证后调用）
        scheduler.step()
        # 早停检查（使用Dice作为监控指标）
        early_stopping(dice_mean, model)
        if dice_mean > best:
            best = dice_mean
            torch.save(model.state_dict(), 'checkpoints/best.pth')
    print(f'训练完成！最佳Dice: {best:.4f}')

if __name__ == '__main__':
    main()