import os, cv2, torch, numpy as np,random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from torch import nn
import torch.nn.functional as F
import unetr_plus_plus.unetr_pp.network_architecture.tumor.unetr_pp_tumor as net
from process import ISIC
from torch.amp import autocast, GradScaler
from  calpixel import calculate_class_weights


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

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0005, restore_best=True):
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
    img_size=256
    batch = 2  
    epochs = 1
    lr = 1e-3
    early_stopping = EarlyStopping()
    num_workers = 4  # 自动适配CPU核心数
    train_ds=ISIC("../ISIC2016",img_size=img_size,mode='train',sample_num=3)
    test_ds=ISIC("../ISIC2016",img_size=img_size,mode='test',sample_num=3)
   
    train_loader = DataLoader(train_ds, batch, shuffle=True, 
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=2  
                            )
    val_loader   = DataLoader(test_ds,   batch, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)

    model = net.UNETR_PP(in_channels=3,
                             out_channels=2,
                             img_size=img_size,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32,64,128,256],
                             do_ds=False,
                             ).to(device)
    class_balance_weights = calculate_class_weights(
        dataset=train_ds,
        device=device,
        weight_type='inverse_frequency'  # 推荐先用这个，极度不平衡再换effective_num
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # 学习率调度器：CosineAnnealingLR（T_max=50，min_lr=1e-5）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, T_max=50, eta_min=1e-5
    )
    loss_fn = DiceCELoss1(device=device,class_balance_weights=class_balance_weights)
    # 初始化混合精度梯度缩放器（新增）
    scaler = torch.amp.GradScaler()  # 关键：用于混合精度训练的梯度缩放

    best = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        if early_stopping.should_stop:
            print(f'早停触发！在第{epoch}轮停止训练')
            break
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for x, y,_ in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            #with torch.amp.autocast(device_type="cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
            # 反向传播：用scaler缩放损失，避免半精度下梯度溢出（新增）
            loss.backward()
            opt.step()
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})


        # 验证
        model.eval()
        preds = []
        gts = []
        dices, ious, recalls, precisions = [], [], [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Val')
            for x, y,_ in val_pbar:
                x, y = x.to(device), y.to(device)
                #with torch.amp.autocast(device_type="cuda"):
                pred = model(x)                            # (B,2,H,W)
                # 四大指标（一行调用）
                dice, iou = metrics_logits(pred, y)
                dices.append(dice)
                ious.append(iou)
               
                # 可选：在进度条上实时显示当前批次的Dice（更直观）
                val_pbar.set_postfix({'batch_dice': f'{dice:.4f}'})
        dice_mean  = np.mean(dices)
        iou_mean   = np.mean(ious)
       

        print(f'Val Dice={dice_mean:.4f} | IoU={iou_mean:.4f} ')
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