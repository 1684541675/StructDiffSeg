import os, cv2, torch, numpy as np,random
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from torch import nn
import torch.nn.functional as F
from diffunet.diffunet_model import DiffUNet
import matplotlib.pyplot as plt
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
    # 1. 模型准备
    model = model.to(device).eval()
    
    # 2. 构造 Dummy Inputs
    # 图像输入
    dummy_image = torch.randn(input_shape, device=device)
    # 标签输入 (必须构造，否则会报错，因为 forward 默认走训练路径需要 gt)
    # 维度要匹配 (B, 1, H, W)，类型要是 float (代码内部会转 long，但输入最好对应 Dataset)
    dummy_gt = torch.zeros((input_shape[0], 1, input_shape[2], input_shape[3]), device=device)

    # ---------- 参数量计算 ----------
    total_params = sum(p.numel() for p in model.parameters())
    train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---------- FLOPs 计算 (核心修改) ----------
    print("正在计算单次前向传播 FLOPs (Paper Metric)...")
    
    # 【关键点】：传入 args=(image, gt)
    # 这会触发 model.forward(image, gt)
    # 这条路径会执行：Encoder(1次) + Decoder(1次)
    # 这正是论文里汇报的 "Model Complexity"
    macs = profile_macs(model, args=(dummy_image, dummy_gt))
    
    flops = macs * 2 # 通常 MACs * 2 = FLOPs
    gflops = flops / 1e9

    print("=" * 50)
    print("Model Complexity (Single Forward Pass)")
    print("(这是你应该写在论文 Table 里的数值)")
    print("=" * 50)
    print(f"Total params : {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Train params : {train_params:,} ({train_params/1e6:.2f} M)")
    print(f"MACs         : {macs/1e9:.2f} G")
    print(f"GFLOPs       : {gflops:.2f} G") # 论文通常汇报这个
    print("=" * 50)

    return total_params, train_params, gflops


# 单独运行参数量计算（无需训练）
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 方式1：仅计算参数量（快速）
    print("仅计算模型参数量...")
    model = DiffUNet(3, 2,ddim_steps=1)
    count_model_parameters(model)
   
    count_model_parameters_and_flops(model)
