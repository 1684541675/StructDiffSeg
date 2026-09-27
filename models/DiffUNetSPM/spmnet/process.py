import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import numpy as np

class ISIC(Dataset):
    def __init__(self, path_Data, img_size=128, train=True, sample_num=None):
        super().__init__()
        self.img_size = img_size
        self.train = train
        self.sample_num = sample_num  # 训练集传500，测试集传150

        # 1. 路径拼接
        if train:
            img_dir = os.path.join(path_Data, 'train', 'images')
            msk_dir = os.path.join(path_Data, 'train', 'masks')
        else:
            img_dir = os.path.join(path_Data, 'test', 'images')
            msk_dir = os.path.join(path_Data, 'test', 'masks')

        # 2. 获取所有路径
        self.images_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        self.masks_list = sorted([f for f in os.listdir(msk_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        all_img_paths = [os.path.join(img_dir, img_name) for img_name in self.images_list]
        all_mask_paths = [os.path.join(msk_dir, msk_name) for msk_name in self.masks_list]

        # 3. 样本筛选（分层抽样）- 修复抽样精度问题
        if self.sample_num is not None and self.sample_num < len(all_img_paths):
            lesion_sizes = []
            for mask_path in all_mask_paths:
                # 增加掩码读取失败处理
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"掩码读取失败：{mask_path}")
                lesion_pct = (mask > 0).sum() / (mask.shape[0] * mask.shape[1])
                if lesion_pct < 0.05:
                    lesion_sizes.append('small')
                elif lesion_pct < 0.2:
                    lesion_sizes.append('medium')
                else:
                    lesion_sizes.append('large')
            # 改用train_size直接指定抽样数量，避免浮点误差
            selected_idx, _ = train_test_split(
                range(len(all_img_paths)),
                train_size=self.sample_num,
                stratify=lesion_sizes,
                random_state=42
            )
            self.img_paths = [all_img_paths[i] for i in selected_idx]
            self.msk_paths = [all_mask_paths[i] for i in selected_idx]
        else:
            self.img_paths = all_img_paths
            self.msk_paths = all_mask_paths
        
        # 计算训练集的均值方差 - 修复硬编码路径问题
        train_img_dir = os.path.join(path_Data, 'train', 'images')
        mean, std = calculate_mean_std(train_img_dir, img_size=img_size)
        # 4. 数据增强 + 归一化（核心修改：添加Resize到增强管道）
        normalize = A.Normalize(mean=mean.tolist(), std=std.tolist(), max_pixel_value=255.0, p=1.0)

        if self.train:
            self.aug = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),  # 图像插值
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    p=0.5, 
                    limit=(-15, 15), 
                    border_mode=cv2.BORDER_REFLECT, 
                    fill=0,
                    fill_mask=0
                ),
                A.GaussianBlur(p=0.2, blur_limit=(3, 3)),
                normalize,  # 归一化：放在增强后、转Tensor前（标准流程）
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
                normalize,  # 测试集也要做相同归一化（关键！）
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取图像和掩码 - 增加读取失败处理
        img_path = self.img_paths[idx]
        msk_path = self.msk_paths[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"图像读取失败：{img_path}")
        img = img[..., ::-1]  # BGR→RGB (H, W, 3)

        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise ValueError(f"掩码读取失败：{msk_path}")

        # 2. 应用数据增强+归一化（包含Resize，无需手动调整尺寸）
        augmented = self.aug(image=img, mask=msk)
        img = augmented['image']  # 归一化后：float32 (3, H, W)
        msk = augmented['mask']   # (H, W) uint8 → Tensor

        # 3. 掩码二值化 - 优化二值化逻辑，提升稳健性
        msk = cv2.threshold(msk.cpu().numpy(), 127, 255, cv2.THRESH_BINARY)[1]  # 先阈值二值化
        msk = torch.from_numpy(msk).unsqueeze(0).float() / 255.0  # (1, H, W)
        msk = (msk > 0.5).float()  # 二值化

        return img, msk, img_path

def calculate_mean_std(img_dir, img_size=256):
    """修正均值方差计算逻辑，增加稳健性处理"""
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    if not img_paths:
        raise ValueError(f"指定目录下未找到有效图像：{img_dir}")
    
    mean = np.zeros(3)
    std = np.zeros(3)
    total_images = 0  # 修正变量名：原total_pixels为计数图像数，非像素数

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"警告：跳过损坏图像 {path}")
            continue
        img = img[..., ::-1]  # BGR→RGB
        # 显式指定插值方式，统一处理
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0  # 先归一化到0-1
        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
        total_images += 1

    if total_images == 0:
        raise ValueError("未找到有效图像用于计算均值方差")
    
    mean /= total_images
    std /= total_images
    return mean, std