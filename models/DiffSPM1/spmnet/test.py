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
from process import ISIC
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_vit = get_r50_b16_config()
    img_size= 256
    # 2. 剩余字段硬编码
    config_vit.n_classes = 2                 # ISIC 二分类
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
    batch=32
    test_ds=ISIC("../ISIC2017",img_size=img_size,train=False)
    
    test_loader = DataLoader(test_ds,   batch, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)
    model = network(3,2,config=config_vit).to(device)
    model.load_state_dict(torch.load('checkpoints/best.pth', map_location=device,weights_only=True))
    # 验证
    model.eval()
    preds = []
    gts = []
    mIoUs, DSCs, Accs, Spes, Sens=[], [], [], [],[]
    inference_times = []  # 存储每个样本的推理时间（秒）
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f'Epoch {1} - test')
        for x, y,_ in test_pbar:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda"):
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
                pred=pred[:, 1:2, :, :]
                pred = torch.sigmoid(pred)
                gts.append(y.squeeze(1).cpu().detach().numpy())
                
                
                pred = pred.squeeze(1).cpu().detach().numpy()
                preds.append(pred) 
                # CosineAnnealingLR每个epoch更新一次学习率（必须在验证后调用）
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=0.5, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    print(
            f'Val | Dice={f1_or_dsc:.4f} | IoU={miou:.4f} | Recall={sensitivity:.4f} | '
            f'Accuracy={accuracy:.4f} | Specificity={specificity:.4f}'
        )
    avg_inference_time = np.mean(inference_times)
    print(f'Average inference time per image: {avg_inference_time:.2f} ms')
    
if __name__ == '__main__':
    main()