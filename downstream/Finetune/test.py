# for test
# python test.py --dataset NIHchest --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth' --device 1
# python test.py --dataset RSNA --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/RSNAPneumonia/popar_cyclic_448_1/best.pth' --device 6
# python test.py --dataset shenzhenCXR --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/shenzhenCXR/popar_cyclic_448_1/best.pth' --device 0

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
from utils import (NativeScalerWithGradNormCount, auto_resume_helper,
                   load_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint)
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
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, vit_base, vit_base_patchsize16, vit_huge_patchsize14,resnet50')
    parser.add_argument('--batch-size', type=int,default=128, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='NIHchest', help="the name of the dataset, eg. CheXpert, NIHchest, shenzhenCXR, RSNA")
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    # parser.add_argument('--pretrain_mode', type=str, default='popar_pec')
    # parser.add_argument('--fold', type=str,default='1', help="10 split of NIHchest dataset")

    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/shenzhenCXR/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/mnt/sda/zhouziyu/liang/rsna-pneumonia-detection-challenge/checkpoints/popar_pec_448_3/best.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default="/sda1/zhouziyu/ssl/downstream_checkpoints/shenzhenCXR/simmim_448_2/best.pth", help='test from checkpoint')
    # parser.add_argument('--resume', default="/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/popar_pec_448_6/best.pth", help='resume from checkpoint')
    
    # parser.add_argument('--output', default='/mnt/sda/zhouziyu/liang/shenzhenCXR/checkpoints/scratch/', type=str, metavar='PATH')
    parser.add_argument('--pretrain_mode', type=str, default=None, help='popar_cyclic,popar_adodocar, popar_pec')
    parser.add_argument('--model_type', type=str,default='swin', help="swin,swinv2")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument("--device", type=str, default='5')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


    args, unparsed = parser.parse_known_args()

    if args.dataset == 'NIHchest':
        config = get_config_NIHchest(args)
    elif args.dataset == 'RSNA':
        config = get_config_RSNA(args)
    elif args.dataset == 'shenzhenCXR':
        config = get_config_shenzhenCXR(args)
    elif args.dataset == 'CheXpert':
        config = get_config_CheXpert(args)
    elif args.dataset == 'vindrcxr':
        config = get_config_vindrcxr(args)
    elif args.dataset == 'SIIM_cls':
        config = get_config_SIIM(args)

    return args, config


def main(config):
    device = torch.device(f'cuda:{config.DEVICE}' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = build_model(config)
    model = model.to(device)
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    # ipdb.set_trace()
    # model.load_state_dict(checkpoint['model'])
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    # val_max_auc = checkpoint['max_auc'] # max_auc
    # print(f'val_max_auc: {val_max_auc}')

    test_dataset, test_loader = build_loader(config, dataset = config.DATA.DATASET)
    sum = 0
    results_path = os.path.dirname(args.resume)
    if os.path.exists(os.path.join(results_path, 'test_result.txt')):
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'a')
    else:
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'w')
    # with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino_pred.txt', 'w') as f:
    for idx, (image, label) in enumerate(test_dataset):
    # for idx, (image, label, imgname) in enumerate(test_dataset):
        if len(image.shape)==3:
            image = torch.unsqueeze(image, dim=0)
        image = image.to(device) # [10,3,224,224]
        
        # print(image.shape)
        # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
        logits = model(image).mean(0).sigmoid() # ncrop
        # print(logits.shape)
        # sys.exit(1)
        logits = logits.cpu().detach().numpy()
        if config.DATA.DATASET == 'RSNA':
            pred = logits.argmax()
            if pred == label.argmax():
                sum+=1

        if idx==0:
            output = logits
            target = label
        else:
            output = np.vstack([output, logits])
            target = np.vstack([target, label])
        print(idx)

        # binary_list = [1 if num >= 0.5 else 0 for num in logits]
        # f.writelines(imgname+' ')
        # f.writelines(str(binary_list)+'\n')
        # print(binary_list)
    # f.close()

    print(f'Test on {config.DATA.TEST_LIST}:', file=log_writer)
    print(output.shape)
    print(target.shape)
    auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
    print(f'auc: {auc}')
    print(f'auc: {auc}', file=log_writer)
    if config.DATA.DATASET == 'CheXpert':
        print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5)
        print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5, file=log_writer)
    print(np.array(auc).mean())
    print(np.array(auc).mean(), file=log_writer)

    if config.DATA.DATASET == 'RSNA':
        acc = sum*1.0/(idx+1)
        print(f'acc: {acc}')
        print(f'acc: {acc}', file=log_writer)



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


if __name__=='__main__':
    args, config = parse_option()
    main(config)