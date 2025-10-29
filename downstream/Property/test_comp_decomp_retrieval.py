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
# from infonce import *
import utils
import vision_transformer as vits
import models.swin_transformer as swins
# from vision_transformer import DINOHead,SimMIM_head,SimMIM_head_SWIN, DenseHead
# from config import config
# from config import update_config
# from config import save_config
# from models import build_model
# from transforms import MultiCropTrainDataTransform, DataAugmentationDINO
# from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
from scipy.stats import ttest_ind
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
from PIL import Image,ImageDraw
import ipdb
from functools import partial


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

    # Misc
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--data_path', default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/', type=str,
        help='Please specify path to the ImageNet training data.')
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




class ChestX_ray14_comp(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, num_class=14, patch_size=448):
        self.img_list = []
        self.img_label = []
        self.augment = transforms.Compose([
                                            # transforms.Resize((224,224)),
                                            transforms.Resize((448,448)),
                                            # Rearrange_and_Norm(),
                                            # torch.from_numpy,
                                           transforms.ToTensor(),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])])
        self.patch_size = patch_size
        #self.transform = transform

        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))



    def random_crop_and_mask(self,image, scale_range=(0.2, 0.5)):
        """
        Randomly crop a part of the image and create a mask of the original image with the
        cropped part hidden.

        Parameters:
            image (PIL.Image): The original image.
            scale_range (tuple): A 2-tuple defining the minimum and maximum scale of the cropped area.

        Returns:
            PIL.Image: Cropped image.
            PIL.Image: Masked image with the cropped area hidden.
        """
        # Get original image size
        orig_width, orig_height = image.size

        # Determine size of the crop
        scale = random.uniform(scale_range[0], scale_range[1])
        crop_width = int(orig_width * scale)
        crop_height = int(orig_height * scale)

        # Determine position of the crop
        left = random.randint(0, orig_width - crop_width)
        upper = random.randint(0, orig_height - crop_height)
        right = left + crop_width
        lower = upper + crop_height

        # Crop the image
        cropped_image = image.crop((left, upper, right, lower))

        # Copy the original image
        masked_image = image.copy()

        # Create a draw object and add a black rectangle over the cropped area
        # Create a draw object and add a black rectangle over the cropped area
        draw = ImageDraw.Draw(masked_image)
        draw.rectangle([left, upper, right, lower], fill=0)


        return cropped_image, masked_image




    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        initial_crop_transform = transforms.RandomResizedCrop(
            1024,  # Final size of the crop
            scale=(0.4, 1),  # Scale range
        )
        origin_image = initial_crop_transform(imageData)



        # Get random crop and masked image
        cropped_image, masked_image = self.random_crop_and_mask(imageData, scale_range=(0.3, 0.8))
        # origin_image.save(os.path.join('./save_image/', f"{index}_origin_image.jpg"))
        # masked_image.save(os.path.join('./save_image/', f"{index}_masked_image.jpg"))
        # cropped_image.save(os.path.join('./save_image/', f"{index}_cropped_image.jpg"))
        # If you have augmentations to apply, you can apply them here
        if self.augment:
            origin_image = self.augment(imageData)
            masked_image = self.augment(masked_image)
            cropped_image = self.augment(cropped_image)


        # Optionally, convert the images to PyTorch tensors here

        return origin_image, masked_image, cropped_image , imageLabel

    def __len__(self):
        return len(self.img_list)



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

    dataset = ChestX_ray14_comp(args.data_path,'./data/xray14/official/test_official.txt')
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

    model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
                        mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)

    # model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    #                     mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #                     drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)
    # checkpoint = torch.load('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0050.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/matrixcompdecompmlp_clstokenglobal_vit_checkpoint0045.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dino_vit_checkpoint0300.pth', map_location='cpu') #checkpoint12N_contrast.pth
    checkpoint = torch.load('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/clstoken_global_vit_ps16/checkpoint.pth', map_location='cpu')
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dropPos_vit-b32_448/droppos.pth', map_location='cpu')
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/byol/checkpoint0300byol.pth', map_location='cpu')
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis_vit/last.pth', map_location='cpu')
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis/last.pth', map_location='cpu')

    

    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
    except:
        # checkpoint = checkpoint['model']
        checkpoint = checkpoint['state_dict']

    
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("module.backbone.", ""): v for k, v in checkpoint_model.items()}

    if 'head.weight' in checkpoint_model:
        del checkpoint_model['head.weight']
    if 'head.bias' in checkpoint_model:
        del checkpoint_model['head.bias']


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

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # student = model
    # student.cuda()
    # embed_dim = student.num_features

    # student = utils.MultiCropWrapper(model, DINOHead(
    #     embed_dim,
    #     args.out_dim,
    #     use_bn=args.use_bn_in_head,
    #     norm_last_layer=args.norm_last_layer,
    # ),DenseHead(),args)

    # # ============ training one epoch of DINO ... ============
    model = model.cuda()
    _ = train_one_epoch(model,data_loader)
    # print(train_stats.shape)

    # save_kde_plot(train_stats.squeeze().tolist(),train_stats_2.squeeze().tolist(), './kde_plotsss_12N_contrast_16.png')
    # # ============ writing logs ... ============

def avg(out):
    return sum(out) / len(out)


def train_one_epoch(model,  data_loader):
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

            spatial_features_origin_image = model.forward_features(origin_image)[:,0] # return the cls token of the input image
            spatial_features_patch = model.forward_features(patch)[:,0]
            spatial_features_masked_image = model.forward_features(masked_image)[:,0]

            # spatial_features_origin_image = model.forward_features(origin_image).mean(dim=1) # return the cls token of the input image
            # spatial_features_patch = model.forward_features(patch).mean(dim=1)
            # spatial_features_masked_image = model.forward_features(masked_image).mean(dim=1)

            # ipdb.set_trace()
            # spatial_features_origin_image = F.softmax(spatial_features_origin_image, dim=-1)
            # spatial_features_patch = F.softmax(spatial_features_patch, dim=-1)
            # spatial_features_masked_image = F.softmax(spatial_features_masked_image, dim=-1)

            subtract_feature = spatial_features_origin_image - spatial_features_masked_image

            
            # Compute similarity between original image and masked image + each patch
            # and keep track of the max similarity and corresponding index
            accuracies_batch = []
            for i in range(spatial_features_origin_image.shape[0]):
                max_similarity = -float('inf')
                max_index = -1

                for j in range(spatial_features_patch.shape[0]):
                    combined_features =  spatial_features_patch[j]#(spatial_features_masked_image[j] +) / 2
                    #print(spatial_features_origin_image[i].cpu().shape, combined_features.cpu().shape)

                    # image retrieval
                    # similarity = compute_similarity(spatial_features_origin_image[i].unsqueeze(0).cpu(), combined_features.unsqueeze(0).cpu())

                    # decomposition
                    similarity = compute_similarity(subtract_feature[i].unsqueeze(0).cpu(), combined_features.unsqueeze(0).cpu())
                    # print(similarity)
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
            print(accuracy)
        print(np.mean(accuracies))


    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    train_dino(args)
