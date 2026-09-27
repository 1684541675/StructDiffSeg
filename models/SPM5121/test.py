import os, cv2, torch, numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
# 把原来 UNet 导入换成 SPM 网络
from vit_seg_configs import get_r50_b16_config   # 我们刚写的 2D 配置
from unet import network                         # 你的 SPM-UNet 2D 版
from process import ISIC,ACDC2DDataset
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix

from train import calculate_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_vit = get_r50_b16_config()
    img_size= 256
    # 2. 剩余字段硬编码
    config_vit.n_classes = 4                 # ISIC 二分类
    config_vit.n_skip = 3
    config_vit.batch_size = 4                # 按 GPU 调
    # 3. 补丁尺寸写死（H=256, W=256, patch=16）
    patch_size = 16
    num_workers = 4  # 自动适配CPU核心数
    h, w = img_size,img_size
    config_vit.patches.grid = (h // patch_size, w // patch_size)
    config_vit.n_patches = (h // patch_size) * (w // patch_size)
    config_vit.h = h // patch_size
    config_vit.w = w // patch_size
    batch=2
    DATA_ROOT = "../ACDC2D"  # 请确认您的路径
    test_ds = ACDC2DDataset(DATA_ROOT, split='test', img_size=img_size)
    
    test_loader = DataLoader(test_ds,   batch, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)
    model = network(1,4,config=config_vit).to(device)
    model.load_state_dict(torch.load('checkpoints/best.pth', map_location=device,weights_only=True))
    # 验证
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
            pred = model(x)                            # (B,2,H,W)
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