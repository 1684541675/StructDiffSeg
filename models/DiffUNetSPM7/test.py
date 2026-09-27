import os, cv2, torch, numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from train import metrics_logits
from diffunet.diffunet_model import DiffUNet
from process import ISIC,ACDC2DDataset1
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from train import calculate_metrics
from spmnet.vit_seg_configs import get_r50_b16_config


@torch.no_grad()
def metrics_vm_unet(logits, target, smooth=1e-7, lesion_class_idx=1):
    """
    对齐VM-UNet的5个核心指标，但聚焦医学分割核心（病灶类），避免背景干扰
    适配场景：二分类医学分割（如ISIC皮肤病变、肿瘤分割，背景类0+病灶类1）
    参考VM-UNet指标定义：mIoU、DSC、Acc、Spe、Sen（返回顺序与VM-UNet表格一致）
    Args:
        logits: (B, C, H, W)  模型输出（二分类时C=2，通道0=背景，1=病灶）
        target: (B, 1, H, W)  真实标签（0/1 mask，0=背景，1=病灶）
        smooth: 数值稳定性参数（避免分母为0，与VM-UNet逻辑一致）
        lesion_class_idx: int  病灶类索引（二分类默认1，标签定义相反可改0）
    Returns:
        mIoU: float  病灶类IoU（非逐类平均，仅聚焦病灶，对应VM-UNet的mIoU(%)）
        DSC: float   病灶类Dice系数（临床核心指标，对应VM-UNet的DSC(%)）
        Acc: float   全局准确率（保留VM-UNet指标，像素级正确比例）
        Spe: float   特异性（防误检，背景类精确率，对应VM-UNet的Spe(%)）
        Sen: float   敏感性（防漏检，病灶类召回率，对应VM-UNet的Sen(%)）
    """
    # 1. 概率计算与预测标签生成（与VM-UNet损失函数逻辑对齐）
    prob = torch.softmax(logits, dim=1)  # (B,C,H,W) → 通道维转概率（0~1）
    pred = prob.argmax(dim=1)            # (B,H,W) → 预测标签（0=背景，1=病灶）

    # 2. 标签格式适配（去除冗余通道，转长整型避免维度错误）
    target = target.squeeze(1).long()  # (B,1,H,W) → (B,H,W)

    # 3. 核心统计量（仅聚焦病灶类TP/FP/FN，同时统计全局TN，适配5个指标）
    # 病灶类关键统计（临床核心：病灶是否分割准确、是否漏检）
    tp = (pred == lesion_class_idx) & (target == lesion_class_idx)  # 病灶真阳性（分割对）
    fp = (pred == lesion_class_idx) & (target != lesion_class_idx)  # 病灶假阳性（背景误判为病灶）
    fn = (pred != lesion_class_idx) & (target == lesion_class_idx)  # 病灶假阴性（病灶漏判为背景）
    # 全局统计（适配Acc和Spe，保留VM-UNet指标完整性）
    tn = (pred != lesion_class_idx) & (target != lesion_class_idx)  # 背景真阴性（背景分割对）
    total_pixels = tp.shape[1] * tp.shape[2]  # 单张图像像素数（H*W）

    # 4. 空间维度求和（每个样本的统计总量 → (B,)）
    tp_sum = tp.float().sum(dim=(1, 2))
    fp_sum = fp.float().sum(dim=(1, 2))
    fn_sum = fn.float().sum(dim=(1, 2))
    tn_sum = tn.float().sum(dim=(1, 2))
    total_sum = total_pixels * torch.ones_like(tp_sum)  # 每个样本的总像素数

    # 5. 计算VM-UNet的5个指标（聚焦病灶，贴合临床，与原返回顺序一致）
    # 5.1 病灶类mIoU（非逐类平均，仅算病灶，临床有意义）
    mIoU = ((tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)).mean()
    # 5.2 病灶类DSC（核心指标，平衡漏检和误检，VM-UNet重点报告项）
    DSC = ((2 * tp_sum + smooth) / (2 * tp_sum + fp_sum + fn_sum + smooth)).mean()
    # 5.3 全局Acc（保留VM-UNet指标，所有像素的预测正确比例）
    Acc = ((tp_sum + tn_sum + smooth) / (total_sum + smooth)).mean()
    # 5.4 Spe（特异性，防误检：背景类不被误判为病灶，VM-UNet要求指标）
    Spe = ((tn_sum + smooth) / (tn_sum + fp_sum + smooth)).mean()
    # 5.5 Sen（敏感性，防漏检：病灶类全部被检出，临床核心需求）
    Sen = ((tp_sum + smooth) / (tp_sum + fn_sum + smooth)).mean()

    # 转numpy scalar返回（方便后续统计，与VM-UNet表格数据格式一致）
    return mIoU.item(), DSC.item(), Acc.item(), Spe.item(), Sen.item()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=2
    img_size=256
    num_workers = 4  # 自动适配CPU核心数
    config_vit = get_r50_b16_config()
    # 2. 剩余字段硬编码
    config_vit.n_classes = 4                 # ISIC 二分类
    config_vit.n_skip = 3
    
    # 3. 补丁尺寸写死（H=256, W=256, patch=16）
    patch_size = 16
    h, w = img_size,img_size
    config_vit.patches.grid = (h // patch_size, w // patch_size)
    config_vit.n_patches = (h // patch_size) * (w // patch_size)
    config_vit.h = h // patch_size
    config_vit.w = w // patch_size
    DATA_ROOT = "../ACDC2D"  # 请确认您的路径
    test_ds = ACDC2DDataset1(DATA_ROOT, split='test', img_size=img_size)
    test_ds.files=test_ds.files[:50]
    
    test_loader = DataLoader(test_ds,   batch, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)
    model = DiffUNet(1, 4,ddim_steps=2,config=config_vit).to(device)
    model.load_state_dict(torch.load('checkpoints/best.pth', map_location=device,weights_only=True))
    model.eval()
    dice_scores = [] # 存每张图的 [RV, Myo, LV] 得分
    inference_times = []  # 存储每个样本的推理时间（秒）
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f'Epoch {1} - test')
        for x, y in test_pbar:
            x, y = x.to(device), y.to(device)
            #with torch.amp.autocast(device_type="cuda"):
            batch_size = x.size(0)  # 获取当前batch的实际样本数（最后一个batch可能不满）
            # ==================== 时间测量开始 ====================
            # 记录推理开始时间（使用cuda事件确保GPU时间精确）
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pred = model(x,ddim=True)                            # (B,2,H,W)
            # 记录推理结束时间
            end_event.record()
            torch.cuda.synchronize()  # 等待GPU操作完成，确保时间准确
            # 计算耗时
            batch_time = start_event.elapsed_time(end_event)/batch_size
            
            inference_times.append(batch_time)
            batch_dices = calculate_metrics(pred, y, n_classes=4)
            dice_scores.append(batch_dices) 
            test_pbar.set_postfix()
            # CosineAnnealingLR每个epoch更新一次学习率（必须在验证后调用）
    dice_scores = np.array(dice_scores) # [Batch数, 3]
    avg_dices = np.mean(dice_scores, axis=0) # [RV平均, Myo平均, LV平均]
    dice_mean = np.mean(avg_dices)           # 最终平均分
    
    print(f"\n[Val] | Mean Dice: {dice_mean:.4f}")
    print(f"      RV: {avg_dices[0]:.4f} | Myo: {avg_dices[1]:.4f} | LV: {avg_dices[2]:.4f}")
        
    avg_inference_time = np.mean(inference_times)
    print(f'Average inference time per image: {avg_inference_time:.2f} ms')
    
if __name__ == '__main__':
    main()