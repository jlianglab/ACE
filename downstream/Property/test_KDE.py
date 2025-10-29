# plot similarity of patch and sub-patch using KDE

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
# import vision_transformer as vits
# import models.swin_transformer as swins
# from vision_transformer import DINOHead,SimMIM_head,SimMIM_head_SWIN, DenseHead
# from config import config
# from config import update_config
# from config import save_config
# from models import build_model
# from models.transforms import DataAugmentationDINO
# from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
from scipy.stats import ttest_ind
from models.swin_transformer_v2 import SwinTransformerV2
# from timm.models.swin_transformer import SwinTransformer
from models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
from random import randint, randrange, choices
from PIL import Image,ImageDraw
import ipdb
from functools import partial
from transformers import AutoModel
import models.convnext as convnext
from models.eva_x import eva_x_base_patch16



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
    parser.add_argument('--batch_size_per_gpu', default=20, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--data_path', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/', type=str,
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


class Rearrange_and_Norm():
    def __call__(self, image):
        # image = cv2.resize(image, (self.size, self.size))
        image = rearrange(image, 'h w c-> c h w')/255
        # image = image/255
        return image

class ChestX_ray14_KDE(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, img_size=448):
        self.img_list = []
        self.img_label = []
        self.img_size = img_size
        self.augment = transforms.Compose([transforms.Resize((img_size,img_size)),
                                            # Rearrange_and_Norm(),
                                            # torch.from_numpy,
                                           transforms.ToTensor(),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])])

        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    # imageLabel = lineItems[1:num_class + 1]
                    # imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    # self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))



    def random_crop_and_mask(self,image, scale_range=(0.2, 0.49)):
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
        # randomly choose the crop number
        # k, l = choices([(1,2), (2,1), (2,2)])[0]
        k, l = 2,2

        # Get original image size
        orig_width, orig_height = image.size

        # Determine size of the crop
        scale = random.uniform(scale_range[0], scale_range[1])
        crop_width = int(orig_width * scale * k)
        crop_height = int(orig_height * scale * l)

        # Determine position of the crop
        # print(orig_width, crop_width, scale, k)
        # print(orig_width - crop_width)
        left = random.randint(0, orig_width - crop_width)
        upper = random.randint(0, orig_height - crop_height)
        right = left + crop_width
        lower = upper + crop_height

        # Crop the image
        whole_crop = image.crop((left, upper, right, lower))
        # whole_crop = np.asarray(whole_crop)

        # sub-crops
        sub_crops = []
        for i in range(k):
            for j in range(l):
                sub_crops.append(image.crop((left+crop_width/k*i, upper+crop_height/l*j, left+crop_width/k*(i+1), upper+crop_height/l*(j+1))))

        return whole_crop, sub_crops

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel = torch.FloatTensor(self.img_label[index])

        initial_crop_transform = transforms.RandomResizedCrop(
            1024,  # Final size of the crop
            scale=(0.4, 1),  # Scale range
        )
        origin_image = initial_crop_transform(imageData)

        # Get random crop and masked image
        whole_crop, sub_crop = self.random_crop_and_mask(origin_image)
        # origin_image.save(os.path.join('./save_image/', f"{index}_origin_image.jpg"))
        # masked_image.save(os.path.join('./save_image/', f"{index}_masked_image.jpg"))
        # cropped_image.save(os.path.join('./save_image/', f"{index}_cropped_image.jpg"))

        whole_crop = self.augment(whole_crop)
        sub_crops = []
        for i in range(len(sub_crop)):
            sub_crops.append(self.augment(sub_crop[i])) 


        # Optionally, convert the images to PyTorch tensors here
        return whole_crop, sub_crops

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



def train_dino(args, backbone='swinv1'):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # transform = DataAugmentationDINO()
    #transform =DataAugmentationDINO()
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    #dataset = ImageFolder_vindr(args.data_path, transform=transform)
    

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    # swinv1
    if backbone == 'swinv1':
        img_size = 448
        model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
        
    
    # swinv2
    elif backbone == 'swinv2':
        model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
        img_size = 512
    
    elif backbone == 'vit_base_patchsize16':
        model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True)
        img_size = 516
    elif backbone == 'convnext':
        model = convnext.__dict__['convnext_base']()
        img_size = 224
    elif backbone == 'eva_x':
        model = eva_x_base_patch16(pretrained = '/mnt/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt') # eva-x
        img_size = 224
    else:
        model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
                                mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)
        img_size = 448
    
    dataset = ChestX_ray14_KDE(args.data_path,'/mnt/nvme1n1/zhouziyu/Swin-Transformer/data/data_split/xray14/official/test_official.txt',img_size=img_size)
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
    
    # checkpoint = torch.load('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0050.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dropPos_vit-b32_448/droppos.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/SelfPatch_vit-b32_448/checkpoint0200.pth', map_location='cpu') #checkpoint12N_contrast.pth
    # checkpoint = torch.load('/mnt/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth', map_location='cpu') # swinv2
    # checkpoint = torch.load('/sda/zhouziyu/ssl/pretrained_model/ACE_v2/large_swinv2_fromIN_unique_multiscale_consis_compdecomp/checkpoint0020.pth', map_location='cpu') # swinv2
    # checkpoint = torch.load('/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar', map_location='cpu') # swinv2
    # checkpoint = torch.load('/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar', map_location='cpu') # swinv2
    # checkpoint = torch.load('/mnt/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth', map_location='cpu') # swinv2
    checkpoint = torch.load('/mnt/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt', map_location='cpu') # swinv2
    

    if not backbone == 'vit_base_patchsize16':
        try:
            checkpoint = checkpoint['student']
        except:
            # checkpoint = checkpoint['model']
            checkpoint = checkpoint # ark, eva-x
            # checkpoint = checkpoint['state_dict']
            # checkpoint = checkpoint['teacher'] # adamv2
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("module.", "backbone."): v for k, v in checkpoint_model.items()}
        

        if 'head.weight' in checkpoint_model:
            del checkpoint_model['head.weight']
        if 'head.bias' in checkpoint_model:
            del checkpoint_model['head.bias']



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

    model = model.cuda()
    _ = train_one_epoch(model,data_loader, backbone)
    # print(train_stats.shape)

    # save_kde_plot(train_stats.squeeze().tolist(),train_stats_2.squeeze().tolist(), './kde_plotsss_12N_contrast_16.png')
    # # ============ writing logs ... ============

def avg(out):
    return sum(out) / len(out)


def train_one_epoch(model, data_loader, backbone='swinv1'):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss =nn.MSELoss()
    similarties_list_1000 = []
    similarties_list = []
    accuracies = []
    
    #torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        for it, (whole_crop, sub_crops) in enumerate(data_loader): # sub_crops list, len()=4
            print(it)
            # print(mask.shape)
            # update weight decay and learning rate according to their schedule
            # if it==1000:
            #     return np.array(similarties_list_1000)
            whole_crop = whole_crop.cuda(non_blocking=True).float()
            for i in range(len(sub_crops)):
                sub_crops[i] = sub_crops[i].cuda(non_blocking=True).float()

            # swin
            if backbone == 'swinv1':
                _, whole_crop_feature = model.forward_features(whole_crop)
                whole_crop_feature = whole_crop_feature.mean(dim=1)
                _, sub_crops_feature = model.forward_features(sub_crops[0])
                sub_crops_feature = sub_crops_feature.mean(dim=1)
                for i in range(1,len(sub_crops)):
                    _, feature=model.forward_features(sub_crops[i])
                    sub_crops_feature+=feature.mean(dim=1)

            # swinv2
            elif backbone == 'swinv2':
                _, whole_crop_feature = model.forward_features(whole_crop)
                whole_crop_feature = whole_crop_feature.mean(dim=1) # return the global embedding of the input image
                _, sub_crops_feature = model.forward_features(sub_crops[0])
                sub_crops_feature = sub_crops_feature.mean(dim=1)
                for i in range(1,len(sub_crops)):
                    _, feature = model.forward_features(sub_crops[i])
                    sub_crops_feature+=feature.mean(dim=1)
                
            # vit
            elif backbone == 'vit_base_patchsize16':
                features = model(whole_crop) # hugging face
                whole_crop_feature = features.last_hidden_state[:,0]
                features = model(sub_crops[0]) # hugging face
                sub_crops_feature = features.last_hidden_state[:,0]
                for i in range(1,len(sub_crops)):
                    features = model(sub_crops[i])
                    sub_crops_feature+=features.last_hidden_state[:,0]
                    
            elif backbone == 'convnext':
                whole_crop_feature = model.extract_features(whole_crop).mean(dim=1)
                sub_crops_feature = model.extract_features(sub_crops[0]).mean(dim=1)
                
                for i in range(1,len(sub_crops)):
                    sub_crops_feature+=model.extract_features(sub_crops[i]).mean(dim=1)

            elif backbone == 'eva_x':
                _, whole_crop_feature = model.forward_features(whole_crop)
                whole_crop_feature = whole_crop_feature[:,1:].mean(dim=1) # return the global embedding of the input image
                _, sub_crops_feature = model.forward_features(sub_crops[0])
                sub_crops_feature = sub_crops_feature[:,1:].mean(dim=1)
                for i in range(1,len(sub_crops)):
                    _, feature = model.forward_features(sub_crops[i])
                    sub_crops_feature+=feature[:,1:].mean(dim=1)
            else:
                whole_crop_feature = model.forward_features(whole_crop)[:,1:].mean(dim=1) # return the global embedding of the input image
                sub_crops_feature = model.forward_features(sub_crops[0])[:,1:].mean(dim=1)
                for i in range(1,len(sub_crops)):
                    sub_crops_feature+=model.forward_features(sub_crops[i])[:,1:].mean(dim=1)



            sub_crops_feature = sub_crops_feature/4

            whole_crop_feature = F.softmax(whole_crop_feature, dim=-1)
            sub_crops_feature = F.softmax(sub_crops_feature, dim=-1)

            
            # Compute similarity between the whole crop and sub-crop
            for i in range(whole_crop_feature.shape[0]):
                # ipdb.set_trace()

                similarity = compute_similarity(whole_crop_feature[i].unsqueeze(0).cpu(), sub_crops_feature[i].unsqueeze(0).cpu())
                # ipdb.set_trace()
                print(similarity)

                similarties_list.append(similarity)
            #print(accuracy)\
        
        print(np.mean(similarties_list))
        similarties_list = np.array(similarties_list)
        np.save('/mnt/nvme1n1/zhouziyu/visualization/KDE_data/eva_x.npy', similarties_list)


    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return 1


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    # args = parser.parse_args()
    
    # train_dino(args, backbone='eva_x')

    


    # plot KDE
    file_path = './KDE_data/KDE.png'
    sim_ace = np.load('./KDE_data/ACEv2_swinv2_large.npy')
    sim_ace = sim_ace.squeeze(1).squeeze(1)

    sim_dino = np.load('./KDE_data/RAD-DINO.npy')
    sim_dino = sim_dino.squeeze(1).squeeze(1)

    sim_droppos = np.load('./KDE_data/eva_x.npy')
    sim_droppos = sim_droppos.squeeze(1).squeeze(1)

    sim_selfpatch = np.load('./KDE_data/Ark6.npy')
    sim_selfpatch = sim_selfpatch.squeeze(1).squeeze(1)


    plt.figure(figsize=(7, 6))
    sns.kdeplot(sim_ace, fill=True,bw_adjust=1,color='hotpink', label='ACE-v2')
    sns.kdeplot(sim_dino, fill=True,bw_adjust=1,color='peachpuff', label='RAD-DINO')
    sns.kdeplot(sim_droppos, fill=True,bw_adjust=1,color='palegreen', label='EVA-X')
    sns.kdeplot(sim_selfpatch, fill=True,bw_adjust=1,color='thistle', label='Ark')
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), borderaxespad=0., prop={'size': 15})
    
    plt.xlim(0.8, 1)
    # current_ticks = np.array([0.42, 0.63, 0.84, 1.05])
    # new_labels = [f'{tick/1.05:.1f}' for tick in current_ticks] # 显示的横坐标除1.2
    # plt.xticks(current_ticks, new_labels)
    
    # plt.title('Feature Similarity')
    plt.xlabel('Feature Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.savefig(file_path)
    # plt.close()
