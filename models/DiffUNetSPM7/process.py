import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import numpy as np

class ISIC(Dataset):
    def __init__(self, path_Data,  img_size=128,mode='train', sample_num=None): # <-- MODIFIED: 'train' boolean replaced with 'mode' string
        super().__init__()
        self.img_size = img_size
        self.mode = mode # <-- MODIFIED
        self.sample_num = sample_num

        # 1. 路径拼接 (MODIFIED to support 'train', 'val', and 'test')
        if self.mode == 'train':
            data_folder = 'train'
        elif self.mode == 'val': # <-- ADDED: Handle validation set
            data_folder = 'val'
        elif self.mode == 'test':
            data_folder = 'test'
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'val', 'test'.")
        
        img_dir = os.path.join(path_Data, data_folder, 'images')
        msk_dir = os.path.join(path_Data, data_folder, 'masks')

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

        # <-- MODIFIED: Augmentation logic now depends on self.mode
        if self.mode == 'train':
            # 训练集使用所有数据增强
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
            # 验证集和测试集只做缩放和归一化
            self.aug = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
                normalize,  # 验证集和测试集也要做相同归一化（关键！）
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
    total_images = 0

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"警告：跳过损坏图像 {path}")
            continue
        img = img[..., ::-1]  # BGR→RGB
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
        total_images += 1

    if total_images == 0:
        raise ValueError("未找到有效图像用于计算均值方差")
    
    mean /= total_images
    std /= total_images
    return mean, std



class ACDC2DDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        self.split = split
        self.img_size = img_size
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.msk_dir = os.path.join(root_dir, split, 'masks')
        
        # 检查路径是否存在
        if not os.path.exists(self.img_dir):
            raise ValueError(f"路径不存在: {self.img_dir}")
            
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        msk_path = os.path.join(self.msk_dir, fname)

        # 读取 (灰度)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # 【关键】强制 Resize 到固定大小，否则 Batch 训练会报错
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # 标签 Resize 必须用最近邻插值 (NEAREST)，防止出现 0.5 这种小数类别
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 归一化 & 维度调整
        image = image.astype(np.float32) / 255.0
        image = image[np.newaxis, :, :]  # [H, W] -> [1, H, W]

        # 标签处理 (确保是 Long 类型，形状 [H, W])
        mask = mask.astype(np.longlong)

        return torch.from_numpy(image), torch.from_numpy(mask)
    
    
    


class ACDC2DDataset1(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        self.split = split
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.msk_dir = os.path.join(root_dir, split, 'masks')
        self.files = sorted(os.listdir(self.img_dir))
        
        # ==================================================
        # 定义增强策略
        # ==================================================
        if split == 'train':
            self.transform = A.Compose([
                # 1. 基础形变：缩放和尺寸调整
                A.Resize(height=img_size, width=img_size),
                
                # 2. 几何变换：模拟位置差异
                A.HorizontalFlip(p=0.5),      # 水平翻转
                A.VerticalFlip(p=0.5),        # 垂直翻转
                A.Rotate(limit=20, p=0.5),    # 随机旋转 +/- 20度
                
                # 3. 弹性形变：医学图像神技 (模拟器官形状变化)
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                
                # 4. 亮度变化：模拟不同机器的扫描亮度差异
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
                
                # 5. 归一化 & 转 Tensor
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            # 验证/测试集：只做 Resize 和 Normalize，绝对不要乱动
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        msk_path = os.path.join(self.msk_dir, fname)

        # 1. 读取数据
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # 2. 应用增强
        # Albumentations 只需要传入 numpy 数组
        augmented = self.transform(image=image, mask=mask)
        
        image = augmented['image'] # 已经是 Tensor [1, H, W] (因为 ToTensorV2)
        mask = augmented['mask']   # 已经是 Tensor [H, W] (注意这里还没处理成 Long)

        # 3. 标签类型转换
        # Albumentations 处理后的 mask 通常是 float 或 int，PyTorch Loss 需要 Long
        mask = mask.long()

        return image, mask


