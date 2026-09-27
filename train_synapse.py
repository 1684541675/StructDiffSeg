import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import os, cv2, torch, numpy as np,random


from datasets.dataset import RandomGenerator,Generator
from engine_synapse import *

#from models.vmunet.vmunet import VMUNet
import models.UNETRPP256.unetr_plus_plus.unetr_pp.network_architecture.tumor.unetr_pp_tumor as unetrpp

from models.SPM5121.unet import network as netspm
from models.SPM5121.vit_seg_configs import get_r50_b16_config

#from models.DiffUNetSPM7.diffunet.diffunet_model import DiffUNet

#from models.DiffUNetSPM.diffunet.diffunet_model import DiffUNet

from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet


import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting_synapse import setting_config

import warnings
warnings.filterwarnings("ignore")


class DoubleConv(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        def down(in_c, out_c): 
            return torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(in_c, out_c))
        def up(in_c, out_c):   
            return torch.nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.inc   = DoubleConv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1   = up(512, 256)
        self.conv1 = DoubleConv(512, 256)
        self.up2   = up(256, 128)
        self.conv2 = DoubleConv(256, 128)
        self.up3   = up(128, 64)
        self.conv3 = DoubleConv(128, 64)
        self.up4   = up(64, 32)
        self.conv4 = DoubleConv(64, 32)
        self.outc  = torch.nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], 1); x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], 1); x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], 1); x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], 1); x = self.conv4(x)
        return self.outc(x)


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()





    print('#----------Preparing dataset----------#')
    train_dataset = config.datasets(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(train_sampler is None),
                                pin_memory=True,
                                num_workers=config.num_workers,
                                sampler=train_sampler)

    
    val_dataset = config.datasets(base_dir=config.val_path, list_dir=config.list_dir, split="val_vol_2d",
                                  transform=transforms.Compose(
                                [Generator(output_size=[config.input_size_h, config.input_size_w])]))
    val_sampler = DistributedSampler(val_dataset, shuffle=True) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(val_sampler is None),
                                pin_memory=True,
                                num_workers=config.num_workers,
                                sampler=val_sampler)
    
    
    
    test_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if config.distributed else None
    test_loader = DataLoader(test_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=test_sampler,
                                drop_last=True)

    
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    
    img_size=256
    #spmnet
    config_vit = get_r50_b16_config()
    
    # 2. 剩余字段硬编码
    config_vit.n_classes = 9                 # ISIC 二分类
    config_vit.n_skip = 3
    config_vit.batch_size = 4                # 按 GPU 调
    # 3. 补丁尺寸写死（H=256, W=256, patch=16）
    patch_size = 16
    h, w = img_size,img_size
    config_vit.patches.grid = (h // patch_size, w // patch_size)
    config_vit.n_patches = (h // patch_size) * (w // patch_size)
    config_vit.h = h // patch_size
    config_vit.w = w // patch_size
    
    
    if config.network == 'vmunet':
        #model=UNet(model_cfg['input_channels'],model_cfg['num_classes'])
        '''
        model = unetrpp.UNETR_PP(in_channels=model_cfg['input_channels'],
                             out_channels=model_cfg['num_classes'],
                             img_size=img_size,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32,64,128,256],
                             do_ds=False,
                             )
        '''
        
        #model = netspm(model_cfg['input_channels'],model_cfg['num_classes'],config=config_vit)
    
        model = DiffUNet(model_cfg['input_channels'],model_cfg['num_classes'],ddim_steps=10)
    
        #model.load_from()
    else: raise('Please prepare a right net!')

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'), weights_only=False)
       
        # 1. 提取权重
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # 2. 静默处理 DataParallel 的 'module.' 前缀不匹配
        is_model_parallel = isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module')
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # 模型要 'module.' 但权重没有 -> 加上
            if is_model_parallel and not k.startswith('module.'):
                new_state_dict['module.' + k] = v
            # 模型不要 'module.' 但权重有 -> 去掉
            elif not is_model_parallel and k.startswith('module.'):
                new_state_dict[k[7:]] = v
            # 一致 -> 保持
            else:
                new_state_dict[k] = v

        # 3. 加载权重 (strict=False 防止因缺少 edge_model 报错，且不打印警告)
        model.load_state_dict(new_state_dict, strict=False)
        
        config.work_dir = config.img_save_path
        epoch=1
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        mean_dice, mean_hd95 = val_one_epoch(
                test_dataset,
                test_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True
            )
        print(mean_dice, mean_hd95)
        return



    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)





    print('#----------Training----------#')
    min_loss = 10e10 
    max_dice=0
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) if config.distributed else None

        loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        val_sampler.set_epoch(epoch) if config.distributed else None

        val_dice = val_one_epoch_2d_dice(val_loader, model, epoch, logger, config)

        # 保存最佳模型 (改为判断 Dice 变大)
        if val_dice > max_dice:
            max_dice = val_dice
            min_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            print(f"✅ 最佳模型更新! Mean Dice: {max_dice:.4f}")

        '''
        loss_val = val_one_epoch_loss(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
        )

        
        
        if loss_val < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss_val
            min_epoch = epoch
            print(f"✅ 最佳验证 Loss 更新: {min_loss:.4f} ")
        
        '''
    
            
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        mean_dice, mean_hd95 = val_one_epoch(
                test_dataset,
                test_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, 
                f'best-epoch{min_epoch}-mean_dice{mean_dice:.4f}-mean_hd95{mean_hd95:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)       