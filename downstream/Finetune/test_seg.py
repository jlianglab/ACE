# for test
# python test_seg.py --seg_part clavicle --resume

import argparse
import datetime
import json
import os
import random
import sys
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
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
from optimizer import build_optimizer
from sklearn.metrics import roc_auc_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from utils import (NativeScalerWithGradNormCount, auto_resume_helper,
                   load_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint)
from models.upernet import UperNet_swin, UperNet_vit, UperNet_resnet50, UperNet_resnet18, UperNet_vit_ps16, UperNet_vit_ps16_new, UperNet_swinv2, UperNet_convnext
from models.unet import UNet
from models.resunet import ModelResUNet_ft
from utils_zzy.dice_loss import DiceLoss
import torch.nn.functional as F
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
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, vit_base')
    parser.add_argument('--batch-size', type=int,default=10, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='JSRT', help="the name of the dataset, eg. JSRT, ChestXdet,SIIM,Montgomery,CovidQuEx")
    parser.add_argument('--pretrain_mode', type=str, default=None, help='popar_pec_seg, seg_simmim,seg_simmim_global,simmim_global_infonce,simmim_global_barlow')
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    parser.add_argument('--seg_part', type=str, default='lung', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/shenzhenCXR/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/rsna-pneumonia-detection-challenge/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default="/sda1/zhouziyu/ssl/downstream_checkpoints/JSRT/seg_simmim_linearprob_448_lung_ratio1003/best.pth", help='test from checkpoint')
    parser.add_argument("--device", type=str, default='7')


    args, unparsed = parser.parse_known_args()

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

    return args, config


def main(config):
    eps = 1e-7
    device = torch.device(f'cuda:{config.DEVICE}' if torch.cuda.is_available() else 'cpu')
    print(device)
    # ipdb.set_trace()
    if config.BACKBONE == 'swin_base':
        model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'swinv2':
        model = UperNet_swinv2(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base':
        model = UperNet_vit(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'resnet50':
        model = UperNet_resnet50(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'resnet18':
        model = UperNet_resnet18(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'unet':
        
        model = UNet(n_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base_patchsize16':
        # ipdb.set_trace()
        if config.PRETRAIN_MODE == 'RAD-DINO':
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, pretrained_path='/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino') 
        else:
            # model = UperNet_vit_ps16(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES) # kad
        
    elif config.BACKBONE == 'resunet':
        model = ModelResUNet_ft(res_base_model='resnet50',out_size=config.MODEL.NUM_CLASSES,imagenet_pretrain = False)
    elif config.BACKBONE == 'convnext':
        model = UperNet_convnext(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    model = model.to(device)
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    # model.load_state_dict(state_dict)
    model.eval()


    test_dataset, test_loader = build_loader(config, dataset = config.DATA.DATASET)

    preds = None
    targets = None

    results_path = os.path.dirname(args.resume)
    if os.path.exists(os.path.join(results_path, 'test_result.txt')):
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'a')
    else:
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'w')

    for idx, (images, target) in enumerate(test_loader):
        target = target.to(device) # [B,C,H,W]
        images = images.to(device) 
        
        with torch.no_grad(): ## 一定要加！！否则显存会崩！
            output = model(images) # [B,C,H,W]
            output = F.sigmoid(output)
            output[output>0.5] =1
            output[output<=0.5] = 0
        
        if preds is None and targets is None:
            preds = output
            targets = target
        else:
            preds = torch.cat((output, preds), 0)
            targets = torch.cat((target, targets), 0)
    
    preds = preds.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    # ipdb.set_trace()
    
    dice_whole_test = dice(preds, targets)
    dice_each_case = mean_dice_coef(preds, targets)

    print(f'Dice for whole test set: {dice_whole_test}')
    print(f'Dice for whole test set: {dice_whole_test}', file=log_writer)
    print(f'Avg dice for each case: {dice_each_case}')
    print(f'Avg dice for each case: {dice_each_case}', file=log_writer)
            



def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(bool)
    im2 = np.asarray(im2 > 0.5).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = np.sum(im1, axis=(0,2,3)) + np.sum(im2, axis=(0,2,3)) # B,C,H,W
    # if im_sum == 0:
    if (im_sum == 0).all():
        return empty_score

    intersection = np.logical_and(im1, im2)
    dice = 2. * np.sum(intersection, axis=(0,2,3)) / im_sum
    print(dice)

    return dice.mean()


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(np.expand_dims(y_true[i,:,:,:],0),np.expand_dims(y_pred[i,:,:,:],0))
    return sum/y_true.shape[0]


if __name__=='__main__':
    args, config = parse_option()
    main(config)