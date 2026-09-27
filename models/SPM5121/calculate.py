import os, cv2, torch, numpy as np
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
from process import ISIC
from torch.amp import autocast, GradScaler
from torchprofile import profile_macs

# ---------- 新增：计算模型参数量函数 ----------
def count_model_parameters(model, verbose=True):
    """
    计算模型的总参数量、可训练参数量和不可训练参数量
    
    Args:
        model: PyTorch模型
        verbose: 是否打印详细信息
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
        non_trainable_params: 不可训练参数量
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()  # 获取当前参数的数量
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
    
    if verbose:
        print("="*50)
        print(f"模型参数量统计")
        print("="*50)
        print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(f"不可训练参数量: {non_trainable_params:,} ({non_trainable_params / 1e6:.2f}M)")
        print(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")
        print("="*50)
    
    return total_params, trainable_params, non_trainable_params

def count_model_parameters_and_flops(model, input_shape=(1, 3, 256, 256), device='cuda'):
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)

    # ---------- 参数量 ----------
    total_params = sum(p.numel() for p in model.parameters())
    train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---------- FLOPs ----------
    # 1) 打开“纯卷积”统计模式
    model._flop_stats_mode = True
    macs = profile_macs(model, dummy)
    flops = macs * 2
    gflops = flops / 1e9
    # 2) 关闭统计模式
    delattr(model, '_flop_stats_mode')

    print("=" * 50)
    print("Model complexity")
    print("=" * 50)
    print(f"Total params : {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Train params : {train_params:,} ({train_params/1e6:.2f} M)")
    print(f"MACs         : {macs/1e9:.2f} G")
    print(f"FLOPs        : {gflops:.2f} GFLOPs")
    print("=" * 50)

    return total_params, train_params, gflops

# 单独运行参数量计算（无需训练）
if __name__ == '__main__':
    # 方式1：仅计算参数量（快速）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_vit = get_r50_b16_config()
    img_size=256
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
    model = network(1,4,config=config_vit)
    print("仅计算模型参数量...")
    count_model_parameters(model)
    
    count_model_parameters_and_flops(model, input_shape=(1, 1, 256, 256), device=device)