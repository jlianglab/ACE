# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# 每次运行修改1.os.environ['MASTER_PORT'] = '12365' 2.output保存路径 3.local_rank指定gpu 4.summarywriter名称
# 合并NIHXraychest14的官方训练和验证集，训练50个epoch

# chestxray
# CUDA_VISIBLE_DEVICES="4,5,6" python -m torch.distributed.launch --nproc_per_node 3 --master_port=25641 poparval_main_ddp.py --img_size 448 --fold 1 --dataset NIHchest
# RSNA
# CUDA_VISIBLE_DEVICES="7" python -m torch.distributed.launch --nproc_per_node 1 --master_port=25645 poparval_main_ddp.py --pretrain_mode 'byol' --fold 1 --dataset RSNA
# shenzhen
# CUDA_VISIBLE_DEVICES="5" python -m torch.distributed.launch --nproc_per_node 1 --master_port=25642 poparval_main_ddp.py --img_size 448 --pretrain_mode 'simmim' --fold 1 --dataset shenzhenCXR

import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
import argparse
import datetime
import json
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from configs.config_NIHchest import get_config_NIHchest
from configs.config_RSNA import get_config_RSNA
from configs.config_shenzhenCXR import get_config_shenzhenCXR
from configs.config_CheXpert import get_config_CheXpert
from configs.config_EyePACS import get_config_eyepacs
from configs.config_vindrcxr import get_config_vindrcxr
from configs.config_SIIM import get_config_SIIM
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from sklearn.metrics import roc_auc_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from utils import (NativeScalerWithGradNormCount, auto_resume_helper,
                   load_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint, save_checkpoint_last)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import ipdb


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/swin/swin_base_patch4_window7_224.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    # easy config modification
    
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, swinv2, vit_base, vit_base_patchsize16, vit_huge_patchsize14, resnet50')
    parser.add_argument('--batch-size', type=int,default=32, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='NIHchest', help="the name of the dataset, eg. NIHchest, shenzhenCXR, RSNA, CheXpert,eyepacs")
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    parser.add_argument('--pretrain_mode', type=str, default='simmim_compose12N_infonce', help='popar_cyclic,popar_adodocar, popar_pec, only_pec, popar, NIHchest, scratch, popar_adar,popar^,popar^_cyclic,popar_cyclic,simmim,simmim_global,simmim_global_infonce,simmim_global_barlow')
    # pretrained mode: comp_decomp, l1loss_local, cosineloss_local, compose_12N, compose_12N_infonce, simmim_compose12N, byol, clip, clip_global, clip_global_simmim
    # contrast_12n_global, simmim_compose12N_infonce,dino,dino_vit,droppos_vit,selfpatch_vit,adam_resnet,contrast_12n_global_inequal_vit,contrast_12n_global_inequal,contrast
    # ijepa_vit, ijepa_ImageNet-1K
    parser.add_argument('--model_type', type=str,default='swin', help="swin,swinv2")
    parser.add_argument('--fold', type=str,default='4', help="10 split of NIHchest dataset")
    parser.add_argument('--pretrain_weight', type=str, default=None)
    # parser.add_argument('--pretrain_weight', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/POPAR/popar_pretrained_weight/simmim/POPAR_swin_depth2,2,18,2_head4,8,16,32_nih14_in_channel3/ckpt_epoch_100.pth')
    parser.add_argument('--mode', type=str, default='train', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--resume', help='resume from checkpoint, 接着训练')
    # parser.add_argument('--output', default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14', type=str, metavar='PATH')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--ratio', type=str, default='100', help='100, 50, 25, 10, 5shot, 10shot')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--master_port', type=str, default='12345')
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--linear_prob',  action='store_true', help='freeze the backbone')
    parser.add_argument('--patience', type=int, default=20, help='stop running if loss does not go down after 20 epoches')

    # args, unparsed = parser.parse_known_args()
    # args = parser.parse_args([])
    args = parser.parse_args()
    if args.dataset == 'NIHchest':
        config = get_config_NIHchest(args)
    elif args.dataset == 'RSNA':
        config = get_config_RSNA(args)
    elif args.dataset == 'shenzhenCXR':
        config = get_config_shenzhenCXR(args)
    elif args.dataset == 'CheXpert':
        config = get_config_CheXpert(args)
    elif args.dataset == 'eyepacs':
        config = get_config_eyepacs(args)
    elif args.dataset == 'vindrcxr':
        config = get_config_vindrcxr(args)
    elif args.dataset == 'SIIM_cls':
        config = get_config_SIIM(args)
    # config = get_config(args)

    return args, config

def main(config, device):
    
    # build_loader写在了data文件夹下__init__.py中
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, dataset = config.DATA.DATASET, ddp=True) # dataloader
    print(f'train set length: {len(dataset_train)}')
    print(f'validation set length: {len(dataset_val)}')

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    # model_keys = list(model.state_dict().keys())
    # print(model_keys)
    # print(len(model_keys))
    # with open('model_keys/rad-dino.txt', 'w') as f:
    #     for i in range(len(model_keys)):
    #         f.writelines(model_keys[i]+'\n')
    # sys.exit(1)

    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.to(device)

    # model = model.to(device)
    model_without_ddp = model

    # linear probing, freeze backbone
    if config.LINEAR_PROB:
        for name, param in model_without_ddp.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = False

    optimizer = build_optimizer(config, model) # TRAIN.OPTIMIZER.NAME = 'adamw'
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else: # config.TRAIN.ACCUMULATION_STEPS = 1
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    if config.DATA.DATASET == 'RSNA':
        criterion = torch.nn.CrossEntropyLoss()

    # max_accuracy = 0.0
    max_auc = 0.0
    max_acc = 0.0

    if config.TRAIN.AUTO_RESUME: # 加载最近训练的模型
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # if config.MODEL.RESUME:
    #     # print(config.MODEL.RESUME)
    #     max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
    #     acc1, acc5, loss = validate(config, data_loader_val, model)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    #     if config.EVAL_MODE:
    #         return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME) and config.MODEL.PRETRAINED!='':
        
        load_pretrained(config, model_without_ddp, logger)
    model_without_ddp = DDP(model_without_ddp, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # 初始化DDP模型

    

        # if config.DATA.DATASET == 'RSNA':
        #     auc, acc, loss = validate(config, data_loader_val, model_without_ddp)

        # else:
        #     auc, loss = validate(config, data_loader_val, model_without_ddp)
        # auc = np.array(auc)
        # avg_auc = auc.mean()
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {avg_auc:.4f}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    # writer = SummaryWriter(log_dir='/data/zhouziyu/ssl/tensorboard_log/'+curr_datetime+config.PRETRAIN_MODE+'_'+str(config.DATA.IMG_SIZE)+'_'+config.DATA.DATASET+config.DATA.FOLD)
    writer = SummaryWriter(log_dir=config.OUTPUT)


    logger.info("Start training")
    start_time = time.time()

    stop_epoch = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        data_loader_train.sampler.set_epoch(epoch)
        # if config.DATA.DATASET != "NIHchest":
        #     data_loader_val.sampler.set_epoch(epoch)

        train_one_epoch(config, model_without_ddp, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, writer)
        
        # if config.DATA.DATASET != 'CheXpert':
        if config.DATA.DATASET == 'RSNA':
            auc, acc, loss = validate(config, data_loader_val, model_without_ddp)
            auc = np.array(auc)
            avg_auc = auc.mean()
            print(avg_auc)
            writer.add_scalar('val/acc', acc, epoch)
        else:
            # if dist.get_rank() == 0:
            auc, loss = validate(config, data_loader_val, model_without_ddp)
            auc = np.array(auc)
            avg_auc = auc.mean()
            print(avg_auc)

            writer.add_scalar('val/avg_auc', avg_auc, epoch)
            writer.add_scalar('val/loss', loss, epoch) # 整个验证集上的平均loss
            if len(auc)==14:
                # logger.info(f"Auc of the network on the {len(dataset_val)} test images: {avg_auc:.3f}")
                writer.add_scalars('val/auc_per_class', {'class1': auc[0],
                                                        'class2': auc[1],
                                                        'class3': auc[2],
                                                        'class4': auc[3],
                                                        'class5': auc[4],
                                                        'class6': auc[5],
                                                        'class7': auc[6],
                                                        'class8': auc[7],
                                                        'class9': auc[8],
                                                        'class10': auc[9],
                                                        'class11': auc[10],
                                                        'class12': auc[11],
                                                        'class13': auc[12],
                                                        'class14': auc[13]}, epoch)
        
        if config.DATA.DATASET == 'RSNA':
            if acc > max_acc:
                stop_epoch = 0
                if dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, avg_auc, optimizer, lr_scheduler, loss_scaler,
                                logger)
            else:
                stop_epoch+=1
            max_acc = max(acc, max_acc)
            max_auc = max(max_auc, avg_auc)

            if dist.get_rank() == 0:
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.3f}")
                logger.info(f'Max acc: {max_acc:.3f}')
        elif config.DATA.DATASET == 'CheXpert':
            avg_5 = (auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5
            writer.add_scalar('val/avg_auc5', avg_5, epoch)
            if avg_5 > max_auc:
                stop_epoch = 0
                save_checkpoint(config, epoch, model_without_ddp, avg_5, optimizer, lr_scheduler, loss_scaler,
                                logger)
            else:
                stop_epoch+=1
            logger.info(f"Auc of the network on the {len(dataset_val)} test images: {avg_5:.3f}")
            max_auc = max(max_auc, avg_5)
            logger.info(f'Max auc: {max_auc:.3f}')
        else:
            if dist.get_rank() == 0:
                # save_checkpoint_last(config, epoch, model_without_ddp, avg_auc, optimizer, lr_scheduler, loss_scaler,
                #                     logger)
                if avg_auc > max_auc:
                    stop_epoch = 0
                    max_auc = avg_auc
                    save_checkpoint(config, epoch, model_without_ddp, avg_auc, optimizer, lr_scheduler, loss_scaler,
                                    logger)
                else:
                    stop_epoch+=1
            
                logger.info(f"Auc of the network on the {len(dataset_val)} test images: {avg_auc:.3f}")
                logger.info(f'Max auc: {max_auc:.3f}')
            
        if stop_epoch>=config.PATIENCE:
            logger.info(f'Evaluation metric has not rise for {stop_epoch} epoches, stop running!')
            sys.exit(1)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

###################################  start  ###############################
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
###################################  end  ###############################

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    # a1 = time.time()
    
    # ###################################  start  ###############################
    # prefetcher = data_prefetcher(data_loader)
    # idx = -1
    # while True:
    #     idx += 1
    #     samples, targets = prefetcher.next()
    #     if samples is None:
    #         break
    # ###################################  end  ###############################

    for idx, (samples, targets) in enumerate(data_loader):
        # b = time.time()
        # read_data = b-a1
        # print(read_data)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # ipdb.set_trace()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            outputs = outputs.to(torch.float32)
        # print('imput_image:', samples.shape)
        # print('output:', outputs.dtype)
        # print('target:', targets.dtype)
        # print(outputs.shape, targets.shape)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS # ACCUMULATION_STEPS=1

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0: # PRINT_FREQ=10
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        # a2 = time.time()
        # iter_time = a2-b
        # print(iter_time)
        writer.add_scalar('train/loss', loss.item(), epoch * num_steps + idx)
        

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.BCEWithLogitsLoss()
    if config.DATA.DATASET == 'RSNA':
        criterion = torch.nn.CrossEntropyLoss()
    # print(criterion)

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    auc_meter = AverageMeter()

    val_target = []
    val_output = []

    end = time.time()
    sum=0
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
            output = output.to(torch.float32)

        # if config.DATA.DATASET == 'RSNA':
        #     target =target.to(torch.int64)
        #     loss = criterion(output, target.squeeze(dim=1))

        loss = criterion(output, target)
        
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        target = target.cpu().numpy()
        output = output.cpu().numpy()
        if idx==0:
            val_target = target
            val_output = output
        else:
            val_target = np.vstack([val_target, target])
            val_output = np.vstack([val_output, output])
        
        if config.DATA.DATASET == 'RSNA':
            pred = output.argmax(axis=1)
            sum+=np.sum((pred == target.argmax(axis=1))!=0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0: # PRINT_FREQ=10
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
    #             f'Test: [{idx}/{len(data_loader)}]\t'
    #             f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
    #             # f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
    #             # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
    #             f'AUC {auc_meter.val:.4f} ({auc_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

    print(val_target.shape)
    print(val_output.shape)
    # print(f'val_target: {val_target}')
    # print(f'val_output: {val_output}')
    auc = metric_AUROC(target = val_target, output = val_output, nb_classes=config.MODEL.NUM_CLASSES)
    print(f'auc: {auc}')
    # auc_meter.update(auc, auc.size[0])
    if len(auc)==14:
        logger.info(f'AUC_class1 {auc[0]:.4f}')
        logger.info(f'AUC_class2 {auc[1]:.4f}')
        logger.info(f'AUC_class3 {auc[2]:.4f}')
        logger.info(f'AUC_class4 {auc[3]:.4f}')
        logger.info(f'AUC_class5 {auc[4]:.4f}')
        logger.info(f'AUC_class6 {auc[5]:.4f}')
        logger.info(f'AUC_class7 {auc[6]:.4f}')
        logger.info(f'AUC_class8 {auc[7]:.4f}')
        logger.info(f'AUC_class9 {auc[8]:.4f}')
        logger.info(f'AUC_class10 {auc[9]:.4f}')
        logger.info(f'AUC_class11 {auc[10]:.4f}')
        logger.info(f'AUC_class12 {auc[11]:.4f}')
        logger.info(f'AUC_class13 {auc[12]:.4f}')
        logger.info(f'AUC_class14 {auc[13]:.4f}')

    if config.DATA.DATASET == 'RSNA':
        acc = sum*1.0/(val_target.shape[0])
        print(f'acc: {acc}')
        return auc, acc, loss_meter.avg
    else:
        return auc, loss_meter.avg



@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    # target = target.cpu().numpy()
    # output = output.cpu().numpy()

    for i in range(nb_classes):
        try:
            outAUROC.append(roc_auc_score(target[:, i], output[:, i]))
        except ValueError:
            pass

    return outAUROC


if __name__ == '__main__':
    args, config = parse_option()

    local_rank = config.LOCAL_RANK
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)


    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = args.master_port

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

   
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, device)
