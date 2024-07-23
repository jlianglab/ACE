# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from infonce import *
import utils
import vision_transformer as vits
import models.swin_transformer as swins
from vision_transformer import DINOHead,SimMIM_head,SimMIM_head_SWIN, DenseHead
from ImageFolder_vindr import ImageFolder_vindr,ChestX_ray14,ShenzhenCXR, ChestX_ray14_comp
from config import config
from config import update_config
from config import save_config
from models import build_model
from transforms import MultiCropTrainDataTransform, DataAugmentationDINO
from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
from scipy.stats import ttest_ind
from timm.models.swin_transformer import SwinTransformer
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'cvt_small', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'vil_14121', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0.8, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=20, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=302, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)")
    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=100, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=5, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--cfg',default='./swin_configs/swin_base_img224_window7.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser






import seaborn as sns
import matplotlib.pyplot as plt

def save_kde_plot(similarities,similarities_2, file_path):
    with open('./simi_result_12N_contrast_16.txt', 'w') as file:
        file.write('\n'.join([str(sim) for sim in similarities]))
    t_stat, p_val = ttest_ind(similarities, similarities_2)
    print(np.array(similarities).mean(),np.array(similarities_2).mean())
    print(f"p_val: {p_val:.30f}")
    sns.kdeplot(similarities, shade=True,bw_adjust=3)
    plt.title('KDE of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.savefig(file_path)
    plt.close()

from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)



def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO()
    #transform =DataAugmentationDINO()
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    #dataset = ImageFolder_vindr(args.data_path, transform=transform)
    dataset = ChestX_ray14_comp(args.data_path,'./data/xray14/official/test_official.txt', augment=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    # model = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
    #                      num_heads=(4, 8, 16, 32), num_classes=3)
    from timm.models.vision_transformer import VisionTransformer, _cfg
    from functools import partial
    import torchvision.models as models
    from timm.models.resnet import resnet50

# # Create the model
# model = (pretrained=True)
    model = resnet50(num_classes=3,features_only=True)
    # model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
    #                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=3)
    state_dict = torch.load('/ocean/projects/med230002p/hluo54/tsne/adam.pth', map_location='cpu') #checkpoint12N_contrast.pth dino_origin/saving_ckpt_CHESTX_corrected_version/dinocheckpoint0300.pth
    # state_dict = modelCheckpoint['model']
    #print(checkpoint.keys())
    # try:
    #     checkpoint = checkpoint['student']
    # except:
    #     checkpoint = checkpoint['state_dict']
    # #checkpoint = checkpoint['student']
    # #print(checkpoint.keys())
    # checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    # checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    # checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    # for key in checkpoint_model.keys():
    #     #print(key)
    #     if key in model.state_dict().keys():
    #         try:
    #             model.state_dict()[key].copy_(checkpoint_model[key])
    #         except:
    #             pass
    #         print("Copying {} <---- {}".format(key, key))
    #     else:
    #         pass

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
    for k in list(state_dict.keys()):
        if k.startswith('fc'):
            del state_dict[k]
    for key in state_dict.keys():
        print(key)
        # 动态地添加 '.backbone' 前缀到键名
        updated_key = key
        #updated_key = f'backbone.{key}'
        if updated_key in model.state_dict().keys():
            model.state_dict()[updated_key].copy_(state_dict[key])
            print("Copying {} <---- {}".format(updated_key, key))
        else:
            print("Key {} is not found".format(updated_key))
    msg = model.load_state_dict(state_dict, strict=False)
    print("=> loaded pretrained model '{}'".format(config.MODEL.PRETRAINED))
    print("missing keys:", msg.missing_keys)

    student = model
    student.cuda()
    # embed_dim = student.num_features

    # student = utils.MultiCropWrapper(model, DINOHead(
    #     embed_dim,
    #     args.out_dim,
    #     use_bn=args.use_bn_in_head,
    #     norm_last_layer=args.norm_last_layer,
    # ),DenseHead(),args)

    # ============ training one epoch of DINO ... ============
    _ = train_one_epoch(student,data_loader)
    # print(train_stats.shape)

    # save_kde_plot(train_stats.squeeze().tolist(),train_stats_2.squeeze().tolist(), './kde_plotsss_12N_contrast_16.png')
    # # ============ writing logs ... ============

def avg(out):
    return sum(out) / len(out)


def train_one_epoch(student,  data_loader):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss =nn.MSELoss()
    similarties_list_1000 = []
    similarties_list_1by1 = []
    accuracies = []
    #torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        for it, (origin_image, masked_image, patch, imageLabel) in enumerate(data_loader):
            # print(mask.shape)
            # update weight decay and learning rate according to their schedule
            # if it==1000:
            #     return np.array(similarties_list_1000)
            origin_image = origin_image.cuda(non_blocking=True).float()
            patch = patch.cuda(non_blocking=True).float() 
            masked_image = masked_image.cuda(non_blocking=True).float()

            spatial_features_origin_image = student.forward(origin_image)[-1].mean(dim=1).reshape(16,-1)
            # print(spatial_features_origin_image[-1].shape)
            #print(spatial_features_origin_image.shape).mean(dim=1)
            spatial_features_patch = student.forward(patch)[-1].mean(dim=1).reshape(16,-1)
            spatial_features_masked_image = student.forward(masked_image)[-1].mean(dim=1).reshape(16,-1)

            print(spatial_features_origin_image.shape)
            # Compute similarity between original image and masked image + each patch
            # and keep track of the max similarity and corresponding index
            accuracies_batch = []
            for i in range(spatial_features_origin_image.shape[0]):
                max_similarity = -float('inf')
                max_index = -1

                for j in range(spatial_features_patch.shape[0]):
                    combined_features =  spatial_features_origin_image[i] #(- spatial_features_masked_image[i] ) / 2
                    #print(spatial_features_origin_image[i].cpu().shape, combined_features.cpu().shape)
                    similarity = compute_similarity(spatial_features_patch[j].unsqueeze(0).cpu(), combined_features.unsqueeze(0).cpu())#spatial_features_origin_image[i].unsqueeze(0).cpu(), combined_features.unsqueeze(0).cpu()
                    #print(similarity)
                    # Update max similarity and corresponding index
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_index = j

                # Compute accuracy
                correct = max_index == i
                # print(correct)
                # print(max_index)
                accuracies_batch.append(correct)

            accuracy = torch.tensor(accuracies_batch).float().mean().item()
            accuracies.append(accuracy)
            #print(accuracy)
        print(np.mean(accuracies))


    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
