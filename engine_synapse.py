import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume

import time

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        #print("images",images.shape) #(1,1,224,224)
        #print("targets",targets.shape) #(1,224,224)
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, pred_edge, uncertainty = model(images, targets)
            uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
            scale = func(epoch, config.epochs)
            loss = criterion(pred, targets,uncertainty,scale)
            loss_edge = criterion(pred_edge, targets)

            loss = loss+loss_edge
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss


def val_one_epoch_loss(val_loader, model, criterion, epoch, logger, config):
    '''
    快速验证函数：只计算 Loss，不生成 3D 结果，不反向传播 (带 tqdm 进度条)
    '''
    stime = time.time()
    model.eval() # 切换到评估模式
    
    loss_list = []
    
    # 初始化 tqdm 进度条
    # desc: 进度条左边的描述文字
    # leave=False: 跑完后进度条消失，保持控制台整洁；如果想保留设为 True
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'[Val] Epoch {epoch}')
    
    with torch.no_grad():
        for iter, data in pbar:
            images, targets = data['image'], data['label']
            images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

            if config.amp:
                with autocast():
                    out = model(images)
                    loss = criterion(out, targets)
            else:
                # 复用训练时的特殊逻辑
                pred, pred_edge, uncertainty = model(images, targets)
                
                uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
                
                # 计算动态权重 scale
                scale = func(epoch, config.epochs) 
                
                loss = criterion(pred, targets, uncertainty, scale)
                loss_edge = criterion(pred_edge, targets)
                loss = loss + loss_edge

            current_loss = loss.item()
            loss_list.append(current_loss)
            
            # 更新进度条右边的信息，实时显示当前 batch 的 loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

    mean_loss = np.mean(loss_list)
    etime = time.time()
    duration = etime - stime
    
    log_info = f'[Fast Val] Epoch {epoch}, Mean Loss: {mean_loss:.4f}, Time: {duration:.2f}s'
    
    # 这里用 print 而不是 tqdm.write，因为进度条 leave=False 已经结束了
    print(log_info) 
    logger.info(log_info)

    return mean_loss

def val_one_epoch_2d_dice(val_loader, model, epoch, logger, config):
    '''
    将 3D 测试的 Dice 计算逻辑应用到 2D 验证中。
    核心思想：Accumulated Dice (全局累计 Dice)
    '''
    model.eval()
    stime = time.time()
    
    # 获取类别数，通常 config.num_classes 或 model_config['num_classes']
    num_classes = config.model_config['num_classes'] 
    
    # --- 核心逻辑移植：初始化全局累加器 ---
    # 对应 3D 代码中：prediction == i 和 label == i 的计算
    # 我们要统计整个验证集每一类的 total_inter (交集) 和 total_union (并集)
    total_inter = torch.zeros(num_classes).cuda()
    total_union = torch.zeros(num_classes).cuda()
    
    # 进度条
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'[Val] Epoch {epoch}')
    
    with torch.no_grad():
        for iter, data in pbar:
            # 1. 数据准备 (和训练一样)
            imgs, labels = data['image'], data['label']
            imgs = imgs.cuda(non_blocking=True).float()
            labels = labels.cuda(non_blocking=True).long() # 标签转 long
            
            # 如果标签是 [B, 1, H, W] 压缩成 [B, H, W]
            if len(labels.shape) == 4:
                labels = labels.squeeze(1)

            # 2. 模型推理 (复用 3D 代码中的推理逻辑)
            if config.amp:
                with autocast():
                    outputs = model(imgs)
            else:
                
                outputs = model(imgs, ddim=True) # 如果你想验证时也用 ddim
               

            # 3. 获取预测结果 (Argmax) -> 对应 3D 代码中的 torch.argmax
            # outputs: [B, num_classes, H, W]
            pred_map = torch.argmax(torch.softmax(outputs, dim=1), dim=1) # [B, H, W]
            
            # 4. --- 核心逻辑移植：计算 Metrics ---
            # 3D 代码是对一个 Volume 循环 slice；这里我们是对一个 Batch 循环
            # 但为了效率，我们直接利用 Tensor 运算对 Batch 并行计算
            
            for c in range(1, num_classes): # 跳过背景类 0
                # 逻辑等同于 metric.binary.dc(pred, gt)
                
                pred_c = (pred_map == c)
                gt_c   = (labels == c)
                
                # 累计交集 (Intersection)
                inter = (pred_c & gt_c).sum()
                
                # 累计并集 (Union = Pred + GT)
                union = pred_c.sum() + gt_c.sum()
                
                total_inter[c] += inter
                total_union[c] += union
            
            # (可选) 进度条显示一点动态，比如当前 batch 是否有前景
            # pbar.set_postfix({'has_fg': (labels>0).any().item()})

    # --- 5. 循环结束，计算最终 Dice ---
    # Dice = 2 * Inter / Union
    # 加上 1e-5 防止除以 0 (如果整个验证集某个器官完全不存在)
    dice_per_class = 2 * total_inter[1:] / (total_union[1:] + 1e-5)
    
    # 计算 Mean Dice (所有类别的平均)
    mean_dice = dice_per_class.mean().item()
    
    # 打印每类的 Dice (可选，方便调试)
    #dice_str = " | ".join([f"Cls{i}:{d:.3f}" for i, d in enumerate(dice_per_class, 1)])
    
    etime = time.time()
    log_info = f'[Val] Epoch {epoch}, Mean Dice: {mean_dice:.4f} , Time: {etime-stime:.2f}s'
    
    print(log_info)
    logger.info(log_info)
    
    return mean_dice

def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95