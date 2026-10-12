import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import os, cv2, torch, numpy as np,random

#from datasets.dataset import RandomGenerator
from engine_synapse import *

from dataset_acdc import *

#from models.vmunet.vmunet import VMUNet
import models.UNETRPP256.unetr_plus_plus.unetr_pp.network_architecture.tumor.unetr_pp_tumor as unetrpp

from models.SPM5121.unet import network as netspm
from models.SPM5121.vit_seg_configs import get_r50_b16_config

#from models.DiffUNetSPM7.diffunet.diffunet_model import DiffUNet

#from models.DiffUNetSPM.diffunet.diffunet_model import DiffUNet
#最优的

from models.DiffUNet3201.diffunet.diffunet_model import DiffUNet

#from models.DiffSPM1.diffunet.diffunet_model import DiffUNet

import os
import sys
import json
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting_acdc import setting_config

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


def run_diffusion_dependence_audit(val_loader, model, config, outputs, logger):
    modes = ('normal', 'zero_state', 'fresh_noise', 'shuffled_state', 'fixed_time')
    output_path = os.path.join(outputs, 'diffusion_dependence_audit.jsonl')
    expected_cases = None
    model.eval()
    with open(output_path, 'w', encoding='utf-8') as record, torch.no_grad():
        for mode in modes:
            set_seed(config.seed)
            rows = []
            case_names = []
            total_inference_seconds = 0.0
            for case_index, data in enumerate(val_loader):
                case_name = data['case_name'][0]
                case_names.append(case_name)
                start = time.perf_counter()
                metric_i = test_single_volume(
                    data['image'], data['label'], model,
                    classes=config.num_classes,
                    patch_size=[config.input_size_h, config.input_size_w],
                    test_save_path=outputs, case=case_name,
                    z_spacing=config.z_spacing, val_or_test=False,
                    ablation_mode=mode, audit_seed=config.seed + case_index * 100000,
                )
                case_seconds = time.perf_counter() - start
                total_inference_seconds += case_seconds
                metrics = np.asarray(metric_i, dtype=float)
                rows.append({
                    'case': case_name,
                    'mode': mode,
                    'rv_dice': float(metrics[0, 0]),
                    'myo_dice': float(metrics[1, 0]),
                    'lv_dice': float(metrics[2, 0]),
                    'mean_dice': float(metrics[:, 0].mean()),
                    'mean_hd95': float(metrics[:, 1].mean()),
                    'case_inference_seconds': case_seconds,
                })
            if expected_cases is None:
                expected_cases = case_names
            elif case_names != expected_cases:
                raise RuntimeError(f'Validation cases changed in ablation mode {mode}')
            for row in rows:
                row['mode_total_inference_seconds'] = total_inference_seconds
                record.write(json.dumps(row, ensure_ascii=False) + '\n')
            record.flush()
            log_info = (f'ACDC diffusion audit: mode={mode}, cases={len(rows)}, '
                        f'total_inference_seconds={total_inference_seconds:.2f}')
            print(log_info)
            logger.info(log_info)
    print(f'ACDC diffusion audit output: {output_path}')
    logger.info('ACDC diffusion audit output: %s', output_path)
    return output_path


def run_image_condition_dominance_audit(val_loader, model, config, outputs, logger):
    modes = ('normal', 'zero_denoiser_image', 'shuffle_denoiser_image',
             'zero_spm_condition')
    output_path = os.path.join(outputs, 'image_condition_dominance_audit.jsonl')
    expected_cases = None
    model.eval()
    with open(output_path, 'w', encoding='utf-8') as record, torch.no_grad():
        for mode in modes:
            set_seed(config.seed)
            rows = []
            case_names = []
            total_inference_seconds = 0.0
            for case_index, data in enumerate(val_loader):
                case_name = data['case_name'][0]
                case_names.append(case_name)
                start = time.perf_counter()
                metric_i = test_single_volume(
                    data['image'], data['label'], model,
                    classes=config.num_classes,
                    patch_size=[config.input_size_h, config.input_size_w],
                    test_save_path=outputs, case=case_name,
                    z_spacing=config.z_spacing, val_or_test=False,
                    condition_mode=mode, audit_seed=config.seed + case_index * 100000,
                )
                case_seconds = time.perf_counter() - start
                total_inference_seconds += case_seconds
                metrics = np.asarray(metric_i, dtype=float)
                rows.append({
                    'case': case_name,
                    'mode': mode,
                    'rv_dice': float(metrics[0, 0]),
                    'myo_dice': float(metrics[1, 0]),
                    'lv_dice': float(metrics[2, 0]),
                    'mean_dice': float(metrics[:, 0].mean()),
                    'mean_hd95': float(metrics[:, 1].mean()),
                    'case_inference_seconds': case_seconds,
                })
            if expected_cases is None:
                expected_cases = case_names
            elif case_names != expected_cases:
                raise RuntimeError(f'Validation cases changed in image condition mode {mode}')
            for row in rows:
                row['mode_total_inference_seconds'] = total_inference_seconds
                record.write(json.dumps(row, ensure_ascii=False) + '\n')
            record.flush()
            log_info = (f'ACDC image condition audit: mode={mode}, cases={len(rows)}, '
                        f'total_inference_seconds={total_inference_seconds:.2f}')
            print(log_info)
            logger.info(log_info)
    print(f'ACDC image condition audit output: {output_path}')
    logger.info('ACDC image condition audit output: %s', output_path)
    return output_path


def run_image_condition_branch_audit(val_loader, model, config, outputs, logger):
    modes = ('normal', 'zero_denoiser_image', 'shuffle_denoiser_image',
             'zero_spm_condition')
    output_path = os.path.join(outputs, 'image_condition_branch_audit.jsonl')
    expected_cases = None
    model.eval()
    with open(output_path, 'w', encoding='utf-8') as record, torch.no_grad():
        for mode in modes:
            set_seed(config.seed)
            rows = []
            case_names = []
            total_inference_seconds = 0.0
            for case_index, data in enumerate(val_loader):
                case_name = data['case_name'][0]
                case_names.append(case_name)
                start = time.perf_counter()
                branches = test_single_volume(
                    data['image'], data['label'], model,
                    classes=config.num_classes,
                    patch_size=[config.input_size_h, config.input_size_w],
                    test_save_path=outputs, case=case_name,
                    z_spacing=config.z_spacing, val_or_test=False,
                    condition_mode=mode, audit_seed=config.seed + case_index * 100000,
                    branch_audit=True,
                )
                case_seconds = time.perf_counter() - start
                total_inference_seconds += case_seconds
                for branch in ('spm', 'diffusion', 'final'):
                    result = branches[branch]
                    metrics = np.asarray(result['metrics'], dtype=float)
                    rows.append({
                        'case': case_name,
                        'condition_mode': mode,
                        'branch': branch,
                        'rv_dice': float(metrics[0, 0]),
                        'myo_dice': float(metrics[1, 0]),
                        'lv_dice': float(metrics[2, 0]),
                        'mean_dice': float(metrics[:, 0].mean()),
                        'mean_hd95': float(metrics[:, 1].mean()),
                        'logit_mean_abs': result['logit_mean_abs'],
                        'logit_std': result['logit_std'],
                        'case_inference_seconds': case_seconds,
                    })
            if expected_cases is None:
                expected_cases = case_names
            elif case_names != expected_cases:
                raise RuntimeError(f'Validation cases changed in branch audit mode {mode}')
            for row in rows:
                row['mode_total_inference_seconds'] = total_inference_seconds
                record.write(json.dumps(row, ensure_ascii=False) + '\n')
            record.flush()
            log_info = (f'ACDC image branch audit: mode={mode}, cases={len(case_names)}, '
                        f'rows={len(rows)}, total_inference_seconds={total_inference_seconds:.2f}')
            print(log_info)
            logger.info(log_info)
    print(f'ACDC image branch audit output: {output_path}')
    logger.info('ACDC image branch audit output: %s', output_path)
    return output_path


def run_spm_embedding_scale_audit(val_loader, model, config, outputs, logger):
    scales = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    output_path = os.path.join(outputs, 'spm_embedding_scale_audit.jsonl')
    expected_cases = None
    model.eval()
    with open(output_path, 'w', encoding='utf-8') as record, torch.no_grad():
        for spm_scale in scales:
            set_seed(config.seed)
            rows = []
            case_names = []
            total_inference_seconds = 0.0
            for case_index, data in enumerate(val_loader):
                case_name = data['case_name'][0]
                case_names.append(case_name)
                start = time.perf_counter()
                branches = test_single_volume(
                    data['image'], data['label'], model,
                    classes=config.num_classes,
                    patch_size=[config.input_size_h, config.input_size_w],
                    test_save_path=outputs, case=case_name,
                    z_spacing=config.z_spacing, val_or_test=False,
                    condition_mode='normal', audit_seed=config.seed + case_index * 100000,
                    branch_audit=True, spm_scale=spm_scale,
                )
                case_seconds = time.perf_counter() - start
                total_inference_seconds += case_seconds
                for branch in ('spm', 'diffusion', 'final'):
                    result = branches[branch]
                    metrics = np.asarray(result['metrics'], dtype=float)
                    rows.append({
                        'case': case_name,
                        'spm_scale': spm_scale,
                        'branch': branch,
                        'rv_dice': float(metrics[0, 0]),
                        'myo_dice': float(metrics[1, 0]),
                        'lv_dice': float(metrics[2, 0]),
                        'mean_dice': float(metrics[:, 0].mean()),
                        'mean_hd95': float(metrics[:, 1].mean()),
                        'logit_mean_abs': result['logit_mean_abs'],
                        'logit_std': result['logit_std'],
                        'case_inference_seconds': case_seconds,
                    })
            if expected_cases is None:
                expected_cases = case_names
            elif case_names != expected_cases:
                raise RuntimeError(f'Validation cases changed at SPM scale {spm_scale}')
            for row in rows:
                row['scale_total_inference_seconds'] = total_inference_seconds
                record.write(json.dumps(row, ensure_ascii=False) + '\n')
            record.flush()
            log_info = (f'ACDC SPM scale audit: scale={spm_scale}, cases={len(case_names)}, '
                        f'rows={len(rows)}, total_inference_seconds={total_inference_seconds:.2f}')
            print(log_info)
            logger.info(log_info)
    print(f'ACDC SPM scale audit output: {output_path}')
    logger.info('ACDC SPM scale audit output: %s', output_path)
    return output_path


def main(config):
    local_debug = os.getenv("LOCAL_DEBUG", "0") == "1"
    trajectory_debug = os.getenv("ACDC_TRAJECTORY_DEBUG", "0") == "1"
    diffusion_audit = os.getenv("ACDC_DIFFUSION_AUDIT", "0") == "1"
    image_condition_audit = os.getenv("ACDC_IMAGE_CONDITION_AUDIT", "0") == "1"
    image_branch_audit = os.getenv("ACDC_IMAGE_BRANCH_AUDIT", "0") == "1"
    spm_scale_audit = os.getenv("ACDC_SPM_SCALE_AUDIT", "0") == "1"
    correction_oracle_audit = os.getenv("ACDC_CORRECTION_ORACLE_AUDIT", "0") == "1"
    disagreement_utility_audit = os.getenv("ACDC_DISAGREEMENT_UTILITY_AUDIT", "0") == "1"
    if ((diffusion_audit or image_condition_audit or image_branch_audit or spm_scale_audit
         or correction_oracle_audit or disagreement_utility_audit) and not config.only_test_and_save_figs):
        raise ValueError("Inference audits require only_test_and_save_figs=True")
    if disagreement_utility_audit and (diffusion_audit or image_condition_audit or image_branch_audit
                                      or spm_scale_audit or correction_oracle_audit
                                      or trajectory_debug or local_debug):
        raise ValueError("Run ACDC_DISAGREEMENT_UTILITY_AUDIT independently of all other audit/debug flags")
    if correction_oracle_audit and (diffusion_audit or image_condition_audit or image_branch_audit
                                    or spm_scale_audit or trajectory_debug or local_debug):
        raise ValueError("Run ACDC_CORRECTION_ORACLE_AUDIT independently of all other audit/debug flags")
    if diffusion_audit and image_condition_audit:
        raise ValueError("Run only one inference audit at a time")
    if image_branch_audit and (diffusion_audit or image_condition_audit or trajectory_debug):
        raise ValueError("Run ACDC_IMAGE_BRANCH_AUDIT independently of other audits/debug flags")
    if spm_scale_audit and (diffusion_audit or image_condition_audit or image_branch_audit
                            or trajectory_debug):
        raise ValueError("Run ACDC_SPM_SCALE_AUDIT independently of other audits/debug flags")

    if local_debug:
        print("⚠️ LOCAL_DEBUG 模式：MX350 本地调试")
        config.batch_size = 1
        config.num_workers = 0
        config.epochs = 1

    
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
    
    root_path="../ACDC"
    img_size=256
    train_dataset = BaseDataSets(base_dir=root_path, split="train", transform=transforms.Compose([
        RandomGenerator([img_size, img_size])]))
    val_dataset = BaseDataSets(base_dir=root_path, split="val")
    test_dataset = BaseDataSets(base_dir=root_path, split="test_vol")
    if trajectory_debug:
        if len(test_dataset) != 40:
            raise ValueError(f"Expected 40 ACDC test cases, found {len(test_dataset)}")
        selected_cases = [name.replace('.h5', '') for name in test_dataset.sample_list]
        print(f"ACDC trajectory debug test cases: {selected_cases}")
        logger.info('ACDC trajectory debug test cases: %s', selected_cases)
    
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(train_sampler is None),
                                pin_memory=True,
                                num_workers=config.num_workers,
                                sampler=train_sampler)

    
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=val_sampler,
                                drop_last=True)
    
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
    
    #spmnet
    config_vit = get_r50_b16_config()
    
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
        #model = DiffUNet(model_cfg['input_channels'],model_cfg['num_classes'],ddim_steps=10,config=config_vit)
        model = DiffUNet(model_cfg['input_channels'],model_cfg['num_classes'],ddim_steps=3)
    
        
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
        if disagreement_utility_audit:
            from disagreement_utility_audit import run_disagreement_utility_audit
            run_disagreement_utility_audit(val_loader, model, config, outputs, logger)
            return
        if correction_oracle_audit:
            from correction_oracle_audit import run_correction_oracle_audit
            run_correction_oracle_audit(val_loader, model, config, outputs, logger)
            return
        if spm_scale_audit:
            run_spm_embedding_scale_audit(val_loader, model, config, outputs, logger)
            return
        if image_branch_audit:
            run_image_condition_branch_audit(val_loader, model, config, outputs, logger)
            return
        if image_condition_audit:
            run_image_condition_dominance_audit(val_loader, model, config, outputs, logger)
            return
        if diffusion_audit:
            run_diffusion_dependence_audit(val_loader, model, config, outputs, logger)
            return
        
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
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)





    print('#----------Training----------#')
    max_dice = 0.0 
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

        
        '''
        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch
        '''
        

        if epoch % config.val_interval == 0:
            mean_dice, mean_hd95 = val_one_epoch(
                    val_dataset,
                    val_loader,
                    model,
                    epoch,
                    logger,
                    config,
                    test_save_path=outputs,
                    val_or_test=False
                )

            # ✅ 新的逻辑：只有当 Dice 分数比之前都高时，才保存 best.pth
            if mean_dice > max_dice:
                max_dice = mean_dice
                min_epoch = epoch # 记录下是哪个 epoch 最强
                
                # 保存模型权重
                save_path = os.path.join(checkpoint_dir, 'best.pth')
                
                torch.save(model.module.state_dict(), save_path)
                
                # 打印一个高亮提示，让你看着爽
                print(f'\n🎉 恭喜！发现新的最佳模型！Epoch: {epoch}, Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}')
                print(f'✅ 模型已保存到: {save_path}\n')
            
            
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
