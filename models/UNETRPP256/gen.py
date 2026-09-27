import os, cv2, torch, numpy as np,csv
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from train import metrics_logits
import unetr_plus_plus.unetr_pp.network_architecture.tumor.unetr_pp_tumor as net
from process import ISIC
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def batch_infer_selected_ids(
    model, 
    model_name, 
    weight_path, 
    data_root, 
    img_size=128, 
    save_mask_root="model_masks"
):
    # ========== 初始化设置 ==========
    
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    num_workers=4
    # 保存mask的文件夹（按模型分类）
    save_mask_dir = os.path.join(save_mask_root, model_name)
    os.makedirs(save_mask_dir, exist_ok=True)

    test_ds=ISIC(data_root,img_size=img_size,train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers,pin_memory=True,
                              prefetch_factor=2)
    # ========== 推理并保存mask ==========
    # mask后处理：转二值图（0/1）
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size))
    ])

    with torch.no_grad():
        for x, y, img_id in test_loader:
            x = x.to(device)
            pred = model(x)  # 模型推理（输出：(B, 2, H, W)，2类：背景/病灶）
            
            # 取病灶类的预测结果（二值化：概率>0.5为1）
            pred_mask = (torch.argmax(pred, dim=1) == 1).float()  # (1, H, W)
            pred_mask_pil = mask_transform(pred_mask.squeeze(0))  # 转PIL图（单通道）
            
            # 保存mask（文件名=图像ID，格式png）
            img_id_basename = os.path.basename(img_id[0]).replace(".jpg", ".png")
            save_path = os.path.join(save_mask_dir, img_id_basename)
            pred_mask_pil.save(save_path)
            print(f"✅ {model_name} 已生成：{save_path}")


# ========== 为每个模型配置并运行（示例） ==========
if __name__ == "__main__":
    # ---------- 通用配置 ----------
    data_root = "../ISIC2017"  # 数据集根路径（ISIC2017文件夹）

    img_size = 256


    # ---------- 1. 你的模型（示例） ----------
    model = net.UNETR_PP(in_channels=3,
                             out_channels=2,
                             img_size=img_size,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32,64,128,256],
                             do_ds=False,
                             ).to(device)
    batch_infer_selected_ids(
        model=model,
        model_name="UNETRPP",  # 对应mask文件夹名
        weight_path="checkpoints/best1.pth",  # 你的模型权重
        data_root=data_root,
        img_size=img_size
    )