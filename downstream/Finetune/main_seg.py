# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# python main_seg.py --seg_part clavicle --fold 1 --local_rank 7 --master_port 12356 --ratio 100 --linear_prob
# python main_seg.py --fold 1 --local_rank 7 --master_port 12356 --dataset SIIM
# python main_seg.py --fold 1 --local_rank 3 --master_port 12354 --dataset Montgomery

# vit_base
# python main_seg.py --backbone vit_base --pretrain_mode vit_seg_selfpatch --pretrain_weight /sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/SelfPatch_vit-b32_448/checkpoint0300.pth --local_rank 6 --dataset SIIM

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
from configs.config_JSRT import get_config_JSRT
from configs.config_ChestXdet import get_config_ChestXdet
from configs.config_SIIM import get_config_SIIM
from configs.config_montgomery import get_config_Montgomery
from configs.config_VindrRib import get_config_VindrRib
from configs.config_CovidQuEx import get_config_ConvidQuEx
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from models.upernet import UperNet_swin, UperNet_vit, UperNet_resnet50, UperNet_vit_ps16, UperNet_resnet18, UperNet_vit_ps16_new, UperNet_swinv2, UperNet_convnext
from models.unet import UNet
from models.resunet import ModelResUNet_ft
from optimizer import build_optimizer
from sklearn.metrics import roc_auc_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from utils import (NativeScalerWithGradNormCount, auto_resume_helper,
                   load_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint)
from utils_zzy.dice_loss import DiceLoss, IoULoss
import torch.nn.functional as F
import math
import ipdb



def parse_option():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/swin/swin_base_patch4_window7_224.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    # easy config modification
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, swinv2, vit_base, resnet50')
    parser.add_argument('--batch-size', type=int,default=16, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--dataset', type=str,default='JSRT', help="JSRT, ChestXdet, SIIM, Montgomery, CovidQuEx")
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    parser.add_argument('--pretrain_mode', type=str, default=None, help='popar_pec_seg, seg_simmim,seg_simmim_global,simmim_global_infonce,simmim_global_barlow')
    # seg_compose_12N, seg_compose_12N_infonce, seg_popar, seg_byol, seg_dino,vit_seg_selfpatch,seg_contrast_12n_global,seg_contrast_12n_global_inequal
    # vit_seg_droppos,vit_seg_dino,seg_contrast_12n_inequal,seg_contrast,adam_resnet_seg,contrast_12n_global_inequal_vit_seg
    parser.add_argument('--fold', type=str,default='1', help="10 split of NIHchest dataset")
    # parser.add_argument('--pretrain_weight', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/POPAR/pretrained_weight/simmim_global_swin_dual_ddp_cover90/ckpt_epoch_100.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/byol/checkpoint0300byol.pth')
    parser.add_argument('--pretrain_weight', type=str, default=None)
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/checkpoint0300_12N_infonce.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/POPAR/popar_pretrained_weight/simmim/POPAR_swin_depth2,2,18,2_head4,8,16,32_nih14_in_channel3/ckpt_epoch_100.pth')

    parser.add_argument('--ratio', type=str, default='100', help='100, 50, 25, 5shot, 10shot, 2shot')
    parser.add_argument('--seg_part', type=str, default='all', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='train', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--resume', help='resume from checkpoint, 接着训练')
    # parser.add_argument('--output', default='/sda1/zhouziyu/ssl/downstream_checkpoints/SIIM', type=str, metavar='PATH')
    parser.add_argument('--linear_prob',  action='store_true', help='freeze the backbone')
    parser.add_argument('--patience', type=int, default=100, help='stop running if loss does not go down after 20 epoches')


    # distributed training
    parser.add_argument("--local-rank", type=int, default=7, help='local rank for DistributedDataParallel')

    parser.add_argument('--master_port', type=str, default='12355')

    # args, unparsed = parser.parse_known_args()
    # args = parser.parse_args([])
    args = parser.parse_args()
    
    if args.dataset == 'JSRT':
        config = get_config_JSRT(args)
    elif args.dataset == 'ChestXdet':
        config = get_config_ChestXdet(args)
    elif args.dataset == 'SIIM':
        config = get_config_SIIM(args)
    elif args.dataset == 'Montgomery':
        config = get_config_Montgomery(args)
    elif args.dataset == 'VindrRib':
        config = get_config_VindrRib(args)
    elif args.dataset == 'CovidQuEx':
        config = get_config_ConvidQuEx(args)
    
    # config = get_config(args)

    return args, config

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, dataset = config.DATA.DATASET) # dataloader
    print(f'train set length: {len(dataset_train)}')
    print(f'validation set length: {len(dataset_val)}')

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    # model = build_model(config)
    # ipdb.set_trace()
    if config.BACKBONE == 'swin_base':
        model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'swinv2':
        model = UperNet_swinv2(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base':
        model = UperNet_vit(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'resnet50':
        model = UperNet_resnet50(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'unet':
        model = UNet(n_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base_patchsize16':
        if config.PRETRAIN_MODE == 'RAD-DINO' or 'DINOv2' in config.PRETRAIN_MODE:
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, pretrained_path='/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino') 
        else:
            # model = UperNet_vit_ps16(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES) # kad
        
    elif config.BACKBONE == 'resnet18':
        model = UperNet_resnet18(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'resunet':
        if config.PRETRAIN_MODE == 'seg_imagenet_resnet50':
            model = ModelResUNet_ft(res_base_model='resnet50',out_size=config.MODEL.NUM_CLASSES,imagenet_pretrain = True)
        else:
            model = ModelResUNet_ft(res_base_model='resnet50',out_size=config.MODEL.NUM_CLASSES,imagenet_pretrain = False)
    elif config.BACKBONE == 'convnext':
        model = UperNet_convnext(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    
    # for k,v in model.state_dict().items():

    # model_keys = list(model.state_dict().keys())
    # # print(model_keys)
    # # print(len(model_keys))
    # with open('model_keys/modelkeys.txt', 'w') as f:
    #     for i in range(len(model_keys)):
    #         f.writelines(model_keys[i]+'\n')
    # sys.exit(1)

    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    # model = model.to(device)
    model_without_ddp = model


    optimizer = build_optimizer(config, model) # TRAIN.OPTIMIZER.NAME = 'adamw'
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    # ipdb.set_trace()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else: # config.TRAIN.ACCUMULATION_STEPS = 1
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    
    if config.DATA.DATASET == 'JSRT':
        if config.SEG_PART == 'all':
            metric = DiceLoss(mode='multilabel')
            criterion = torch.nn.CrossEntropyLoss()
        else:
            metric = DiceLoss(mode='binary')
            criterion = torch.nn.BCEWithLogitsLoss()
    elif config.DATA.DATASET == 'VindrRib':
        metric = DiceLoss(mode='multilabel')
        criterion = torch.nn.BCEWithLogitsLoss()
    else: 
        # metric = IoULoss()
        metric = DiceLoss(mode='binary')
        criterion = torch.nn.BCEWithLogitsLoss()

    max_dice = 0

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


    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)


    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    # writer = SummaryWriter(log_dir='/data/zhouziyu/ssl/tensorboard_log/'+curr_datetime+config.PRETRAIN_MODE+'_'+str(config.DATA.IMG_SIZE)+'_'+config.DATA.DATASET+config.DATA.FOLD)
    writer = SummaryWriter(log_dir=config.OUTPUT)


    logger.info("Start training")
    start_time = time.time()

    # linear probing, freeze backbone
    if config.LINEAR_PROB:
        for param in model_without_ddp.backbone.parameters():
            param.requires_grad = False

    stop_epoch = 0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config, model_without_ddp, criterion, metric, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, writer)
        

        dice_score, loss = validate(config, data_loader_val, model_without_ddp, criterion, metric)


        writer.add_scalar('val/dice', dice_score, epoch)
        writer.add_scalar('val/loss', loss, epoch) # 整个验证集上的平均loss

        
        if dice_score > max_dice:
            stop_epoch = 0
            save_checkpoint(config, epoch, model_without_ddp, dice_score, optimizer, lr_scheduler, loss_scaler,
                        logger)
        else:
            stop_epoch+=1
        max_dice = max(max_dice, dice_score)

        logger.info(f"Dice of the network on the {len(dataset_val)} test images: {dice_score:.3f}")
        logger.info(f'Max dice: {max_dice:.3f}')

        if stop_epoch>=config.PATIENCE:
            logger.info(f'Val dice has not rise for {stop_epoch} epoches, stop running!')
            sys.exit(1)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



def train_one_epoch(config, model, criterion, metric, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    ce_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()


    for idx, (samples, targets) in enumerate(data_loader):
        # b = time.time()
        # read_data = b-a1
        # print(read_data)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            outputs = outputs.to(torch.float32)

        # ipdb.set_trace()
        dice_loss = metric(outputs, targets)

        # if config.DATA.DATASET == 'ChestXdet':
        #     dice_loss = (2*dice_loss)/(1+dice_loss) # diceloss转为iouloss

        dice_meter.update(dice_loss.item(), targets.size(0))
        ce_loss = criterion(outputs, targets)
        ce_meter.update(ce_loss.item(), targets.size(0))


        loss = ce_loss + dice_loss
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
        # nan
        if math.isnan(grad_norm):
            print(outputs)
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
                f'dice loss {dice_meter.val:.4f} ({dice_meter.avg:.4f})\t'
                f'ce loss {ce_meter.val:.4f} ({ce_meter.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        # a2 = time.time()
        # iter_time = a2-b
        # print(iter_time)
        writer.add_scalar('train/loss', loss.item(), epoch * num_steps + idx)
        writer.add_scalar('train/dice_loss', dice_loss.item(), epoch * num_steps + idx)
        

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion, metric):

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    end = time.time()
    sum=0
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
            output = output.to(torch.float32)

        loss = criterion(output, target)
        dice_loss = metric(output, target)
        if math.isnan(dice_loss):
            print(output)
        # if config.DATA.DATASET == 'ChestXdet':
        #     dice_loss = (2*dice_loss)/(1+dice_loss) # diceloss转为iouloss
        
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        dice_meter.update(dice_loss.item(), target.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0: # PRINT_FREQ=10
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'CE Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Dice loss {dice_meter.val:.4f} ({dice_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

    return 1-dice_meter.avg, loss_meter.avg





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

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    rank = 0 # 全局的进程id？
    world_size = 1 # 某个节点上的进程数，一个进程可以对应若干个gpu，一般一个进程使用一块gpu，这种情况下world_size等于所有节点的gpu数量
    
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    
    # torch.distributed.init_process_group('gloo', init_method='file://tmp/somefile', rank=0, world_size=1)
    torch.distributed.barrier()

    # seed = config.SEED + dist.get_rank()
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True
    print(dist.get_rank())
    
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

    # ipdb.set_trace()
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

    main(config)
