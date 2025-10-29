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
from configs.config_NIHchest_keypoint import get_config_NIHchest_keypoint
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
from models.upernet import UperNet_swin, UperNet_vit, UperNet_swinv2, UperNet_vit_ps16_new
from utils_zzy.dice_loss import DiceLoss
import torch.nn.functional as F
from utils_zzy.evaluation import compute_error, get_preds, draw_points_and_save_images, save_images_with_heatmaps, save_images_with_heatmaps_points
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
    parser.add_argument('--dataset', type=str,default='NIHchest_keypoint', help="the name of the dataset, eg. NIHchest_keypoint")
    parser.add_argument('--pretrain_mode', type=str, default=None, help='popar_pec_seg, seg_simmim,seg_simmim_global,simmim_global_infonce,simmim_global_barlow')
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    parser.add_argument('--seg_part', type=str, default='lung', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')
    parser.add_argument('--model_type', type=str,default='swin', help="swin,swinv2")
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--select_index', type=list, default=[2,10,18,34,42,50,21], help='selected indexes of the landmarks. eg.[2,10,18,34,42,50,21],[2,34,29,24,40,10,30,25,46,16,32,54,53]')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/shenzhenCXR/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/rsna-pneumonia-detection-challenge/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default="/sda1/zhouziyu/ssl/downstream_checkpoints/JSRT/seg_simmim_linearprob_448_lung_ratio1003/best.pth", help='test from checkpoint')
    parser.add_argument("--device", type=str, default='4')


    args, unparsed = parser.parse_known_args()

    if args.dataset == 'NIHchest_keypoint':
        config = get_config_NIHchest_keypoint(args)

    return args, config


def main(config):
    save_dir = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/keypoint/contrast_12n_global_inequalswin_base_448_heatmap_13points'
    save_dir2 = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/keypoint/contrast_12n_global_inequalswin_base_448_13points'
    eps = 1e-7
    device = torch.device(f'cuda:{config.DEVICE}' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    if config.BACKBONE == 'swin_base':
        # if config.MODEL.TYPE == 'swin':
        model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'swinv2':
        model = UperNet_swinv2(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base':
        model = UperNet_vit(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'resnet50':
        model = UperNet_resnet(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    elif config.BACKBONE == 'vit_base_patchsize16':
        # ipdb.set_trace()
        if config.PRETRAIN_MODE == 'RAD-DINO':
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, pretrained_path='/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino') 
        else:
            # model = UperNet_vit_ps16(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
            model = UperNet_vit_ps16_new(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES) # kad
    model = model.to(device)
    # ipdb.set_trace()
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    # model.load_state_dict(state_dict)
    model.eval()


    test_dataset, test_loader = build_loader(config, dataset = config.DATA.DATASET)

    preds = None
    targets = None
    error_meter = AverageMeter() # keypoint predict error

    results_path = os.path.dirname(args.resume)
    if os.path.exists(os.path.join(results_path, 'test_result.txt')):
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'a')
    else:
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'w')

    for idx, (image_init, images, target, tcoords, image_name) in enumerate(test_loader):
        # print(image_init.shape) # [B,C,H,W]
        target = target.to(device) # [B,C,H,W]
        images = images.to(device) 
        
        with torch.no_grad(): ## 一定要加！！否则显存会崩！
            output = model(images) # [B,C,H,W]
            output = F.sigmoid(output)
            error = compute_error(output, tcoords)
            # ipdb.set_trace()
            error = error*1024/config.DATA.IMG_SIZE # 1024/448
            print(error.mean().item())
            error_meter.update(error.mean().item(), target.size(0))

            pcoords = get_preds(output) # B,7,2
            # draw_points_and_save_images(image_init, tcoords, pcoords, save_dir2, image_name)
            # # save_images_with_heatmaps(image_init, output, save_dir, image_name)
            # save_images_with_heatmaps_points(image_init, output, tcoords, pcoords, save_dir, image_name)

    
    print(error_meter.avg)
    print(f'Avg pixel error: {error_meter.avg}', file=log_writer)
            






if __name__=='__main__':
    args, config = parse_option()
    main(config)