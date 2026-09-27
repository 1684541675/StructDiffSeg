import torch
import numpy as np
from process import ISIC  # 导入你的ISIC数据集类

def calculate_class_weights(dataset, device='cuda', weight_type='inverse_frequency', beta=0.9999):
    """
    计算二分类分割任务的类别平衡权重（背景0/病灶1）
    Args:
        dataset: 实例化的ISIC数据集对象（已加载mask标签）
        device: 权重最终要部署的设备（cuda/cpu）
        weight_type: 权重计算方式
                     - inverse_frequency: 逆频率权重（基础版）
                     - effective_num: 有效样本数权重（更鲁棒，适合类别极度不平衡）
        beta: 有效样本数权重的超参数（通常取0.9999）
    Returns:
        class_balance_weights: 形状为[2]的torch张量（背景权重，病灶权重），已移至指定设备
    """
    # 1. 初始化类别像素计数器
    class_counts = np.zeros(2, dtype=np.float64)  # [背景0的像素数, 病灶1的像素数]
    
    # 2. 遍历数据集，统计每个类别的总像素数
    #print("正在统计数据集类别像素分布...")
    for idx in range(len(dataset)):
        # 取出单个样本的mask（形状：(1, H, W)）
        _, mask, _ = dataset[idx]  # ISIC数据集返回 (img, mask, name)
        mask = mask.squeeze(0).numpy()  # 压缩维度：(H, W)
        
        # 统计0/1的像素数并累加
        class_counts[0] += np.sum(mask == 0)  # 背景像素数
        class_counts[1] += np.sum(mask == 1)  # 病灶像素数
    
    # 3. 防止除零（极端情况某类无像素）
    eps = 1e-8
    class_counts = class_counts + eps
    
    # 4. 计算类别权重
    total_pixels = class_counts.sum()
    n_classes = 2
    
    if weight_type == 'inverse_frequency':
        # 逆频率权重（最常用）：weight = 总像素数 / (类别数 * 该类像素数)
        weights = total_pixels / (n_classes * class_counts)
    
    elif weight_type == 'effective_num':
        # 有效样本数权重（适合极度不平衡场景）：weight = (1 - beta) / (1 - beta^count)
        weights = (1 - beta) / (1 - np.power(beta, class_counts))
    
    else:
        raise ValueError(f"不支持的权重类型：{weight_type}，可选：inverse_frequency/effective_num")
    
    # 5. 归一化权重（可选，确保权重和为1，不影响CrossEntropyLoss效果）
    weights = weights / weights.sum() # 保持权重的相对比例，和为类别数
    
    # 6. 转换为torch张量并移至目标设备
    class_balance_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    '''
    # 打印统计信息
    print("="*50)
    print(f"类别像素统计：")
    print(f"背景(0)像素数：{class_counts[0]:.0f} ({class_counts[0]/total_pixels*100:.2f}%)")
    print(f"病灶(1)像素数：{class_counts[1]:.0f} ({class_counts[1]/total_pixels*100:.2f}%)")
    print(f"\n计算得到的类别权重：")
    print(f"背景(0)权重：{class_balance_weights[0].item():.4f}")
    print(f"病灶(1)权重：{class_balance_weights[1].item():.4f}")
    print("="*50)
    '''
    
    
    return class_balance_weights

# ------------------- 执行计算 -------------------
if __name__ == '__main__':
    # 固定随机种子（可选）
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 实例化ISIC数据集（和训练代码保持一致的img_size）
    img_size = 256
    train_ds = ISIC("../ISIC2017", img_size=img_size, train=True,sample_num=210)
    
    # 2. 计算类别权重（可选：inverse_frequency / effective_num）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_balance_weights = calculate_class_weights(
        dataset=train_ds,
        device=device,
        weight_type='inverse_frequency'  # 推荐先用这个，极度不平衡再换effective_num
    )
    
    # 3. 输出最终可直接代入DiceCELoss的权重
    print(f"\n最终可直接使用的权重张量：{class_balance_weights}")