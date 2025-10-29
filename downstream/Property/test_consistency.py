# visualize the t-sne of the ground-truth features and synthetic features using interpolation

# PEAC and ACE t-SNE code

import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from torch.nn import AdaptiveAvgPool2d
# from timm.models.swin_transformer import SwinTransformer
from models.swin_transformer import SwinTransformer
from models.swin_transformer_v2 import SwinTransformerV2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image, ImageOps
import torch.nn as nn
from sklearn.cluster import KMeans
import shutil
import sys
import models.convnext as convnext
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import ipdb
import csv
import seaborn as sns
from scipy.spatial.distance import cdist


def apply_gaussian_kernel(kernel, point, matrix_size=14):
    half_k = kernel.shape[0] // 2
    # matrix = np.ones((matrix_size, matrix_size))
    
    x, y = point
    result_coords = []
    result_weights = []
    
    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            if 0 <= x + i < matrix_size and 0 <= y + j < matrix_size:
                result_coords.append((x + i, y + j))
                result_weights.append(kernel[half_k + i, half_k + j])
    
    return result_coords, result_weights


def gaussian_kernel_normalized(size, sigma=1):
    """Generates a (size x size) Gaussian kernel with mean 0 and standard deviation sigma,
    normalized so that the center value is 1."""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = g / g.sum()  # Normalize to sum to 1
    g = g / g[size//2, size//2]  # Normalize so the center value is 1
    return g


def get_patch_matching_target_1to1(overlap_mask_1, overlap_mask_2, patch_num=14):
    """
    input: 
     overlap_mask_1: overlap index of crop1 (bool, [196,])
     overlap_mask_2: overlap index of crop2 (bool, [196,])

    output: two target matrices of matrix matching, size 196*196

    """
    

    # find True in overlap_mask_2
    true_indices_2 = torch.nonzero(overlap_mask_2).squeeze() # crop2: overlap patches:n
    true_indices_1 = torch.nonzero(overlap_mask_1).squeeze() # crop1: overlap patches:n


    # initialize the target of matching matrix
    target_matrix = torch.zeros(patch_num*patch_num, patch_num*patch_num)

    # get 5*5 gaussian kernel
    # kernel = gaussian_kernel_normalized(size=7)
    kernel = gaussian_kernel_normalized(size=13)


    for i in range(len(true_indices_2)): 
        idx2 = true_indices_2[i] # index in crop2 0~195
        idx1 = true_indices_1[i] # index in upsample crop1 0~783

        row, col = divmod(idx1.item(), patch_num)

        coords, weights = apply_gaussian_kernel(kernel, (row, col), matrix_size=patch_num) # apply 5*5 gaussion weights to the matrx matching target

        for j in range(len(coords)):
            corresponding_idx = coords[j][0] * patch_num + coords[j][1]

            target_matrix[idx2,corresponding_idx] = weights[j]
    
    return  target_matrix # [196,196]



def get_index(a, b, c=(1,1), patch_num=14): 
# 输入：a为crop1左上角grid的index，b为patch2左上角grid的index
# 输出：随机挑选出的crop1和crop2对应patch的索引
    """
    get the overlap mask of crop1 and crop2
    input:
      a: crop1's top left corner index, 
      b: crop2's top left corner index,
      c: the h, w rate of crop1, (14*k)*(14*l)
    output: the overlap mask of crop1 and crop2, and the shape of overlap masks are all 14*14
    """
    (idx_x1, idx_y1), (idx_x2, idx_y2), (k, l) = a, b, c
    # print(a,b,c)
    # 重合部分index范围
    idx_xmin, idx_xmax = max(idx_x1, idx_x2), min((idx_x1+patch_num*l), (idx_x2+patch_num))
    idx_ymin, idx_ymax = max(idx_y1, idx_y2), min((idx_y1+patch_num*k), (idx_y2+patch_num))

    # 找出重合部分在crop1中对应的index list
    overlap_mask_1 = torch.zeros((patch_num,patch_num))
    overlap_mask_1[(idx_ymin-idx_y1)//k : (idx_ymax-idx_y1)//k,(idx_xmin-idx_x1)//l : (idx_xmax-idx_x1)//l] = 1
    overlap_mask_1 = overlap_mask_1.flatten()
    index1 = torch.nonzero(overlap_mask_1)
    # print(index1.shape)

    overlap_mask_2 = torch.zeros((patch_num,patch_num))
    overlap_mask_2[(idx_ymin-idx_y2)//k*k:(idx_ymax-idx_y2)//k*k,(idx_xmin-idx_x2)//l*l:(idx_xmax-idx_x2)//l*l] = 1
    overlap_mask_2 = overlap_mask_2.flatten()
    index2 = torch.nonzero(overlap_mask_2)
    # print(index2.shape)

    return overlap_mask_1.bool(), overlap_mask_2.bool()



def generate_heatmap_from_patch_distance(c1_start, c2_start, grid_size=14):
    """
    c1_start: (row, col) of crop 1 starting point in 32x32 grid
    c2_start: (row, col) of crop 2 starting point in 32x32 grid
    """

    # Create 14x14 grid offsets
    dx = np.arange(grid_size)
    dy = np.arange(grid_size)
    x, y = np.meshgrid(dx, dy)

    # Get (x, y) coordinates relative to the start point
    c1_coords = np.stack([x + c1_start[1], y + c1_start[0]], axis=-1).reshape(-1, 2)  # shape: (196, 2)
    c2_coords = np.stack([x + c2_start[1], y + c2_start[0]], axis=-1).reshape(-1, 2)

    # Compute pairwise distances (Euclidean)
    heatmap = cdist(c1_coords, c2_coords, metric='euclidean')  # shape: (196, 196)

    return heatmap


def normalize_heatmap(heatmap):
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    return (heatmap - min_val) / (max_val - min_val + 1e-8)



def tsne(device, model, model_path="./POC_R_T_L.pth",save_name="tsne_plot",flip=0, args=None):
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['teacher']
    except:
        # checkpoint = checkpoint['model']
        checkpoint = checkpoint['state_dict']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    model.cuda()
    
    img = cv2.imread('/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/00000001_000.png')
    
    C1 = (5,8)
    C2 = (11, 14)
    C1_start = (C1[0]*32, C1[1]*32)
    C2_start = (C2[0]*32, C2[1]*32)
    
    patch1 = img[C1_start[1]:C1_start[1]+args.img_size, C1_start[0]:C1_start[0]+args.img_size]
    patch2 = img[C2_start[1]:C2_start[1]+args.img_size, C2_start[0]:C2_start[0]+args.img_size]
    
    patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB)
    patch1 = Image.fromarray(patch1)
    patch1 = transform(patch1).unsqueeze(0).to(device)
    
    patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
    patch2 = Image.fromarray(patch2)
    patch2 = transform(patch2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, features1 = model.forward_features(patch1) # swin:[1,196,768] vit:[1,197,768]
        _, features2 = model.forward_features(patch2) # swin:[1,196,768] vit:[1,197,768]

    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    f1 = features1[0]
    f2 = features2[0]
    
    # 对每个向量进行 L2 归一化
    f1_norm = f1 / np.linalg.norm(f1, axis=1, keepdims=True)
    f2_norm = f2 / np.linalg.norm(f2, axis=1, keepdims=True)
    
    cosine_similarity = np.dot(f1_norm, f2_norm.T)
    
    
    # 可视化互相关矩阵
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cosine_similarity, cmap='viridis', vmin=0, vmax=1)
    plt.title("Cosine Similarity Matrix (0-1 Scaled)")
    plt.xlabel("Feature Index in Map 2")
    plt.ylabel("Feature Index in Map 1")
    # 设置横纵坐标间隔为16
    xticks = list(range(0, cosine_similarity.shape[1], 16))
    yticks = list(range(0, cosine_similarity.shape[0], 16))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_name+"cosine_similarity_matrix.png")


    # gt matrix
    sample_index1, sample_index2 = get_index(C2, C1, c=(1,1), patch_num=16)
    target_matrix = get_patch_matching_target_1to1(sample_index1, sample_index2, patch_num=16)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(target_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.title("Ground Truth Similarity Matrix (0-1 Scaled)")
    plt.xlabel("Feature Index in Map 2")
    plt.ylabel("Feature Index in Map 1")
    # 设置横纵坐标间隔为16
    xticks = list(range(0, target_matrix.shape[1], 16))
    yticks = list(range(0, target_matrix.shape[0], 16))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_name+"gt_similarity_matrix.png")
    
    
    # distance matrix
    heatmap = generate_heatmap_from_patch_distance(C1, C2, grid_size=16)
    heatmap_norm = normalize_heatmap(heatmap)
    # plt.imshow(heatmap_norm, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Euclidean Distance')
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_norm, cmap='viridis', vmin=0, vmax=1)
    plt.title('Distance Heatmap Between C1 and C2')
    plt.xlabel('C2 grid index')
    plt.ylabel('C1 grid index')
    # 设置横纵坐标间隔为16
    xticks = list(range(0, heatmap_norm.shape[1], 16))
    yticks = list(range(0, heatmap_norm.shape[0], 16))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    plt.tight_layout()
    plt.savefig(save_name+'distance_matrix.png')

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the properties of interpolation, extrapolation and triangulation.')
    parser.add_argument('--image_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/sslgenesis/fromscratch_extrap_shuffle_compdecomp_consis/checkpoint.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/consistency/pretrained_weight/consis/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/ACE_v2/large_swinv2_fromIN_unique_multiscale_consis_compdecomp/checkpoint0020.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/simmim/ckpt_epoch_100.pth',  help='The root dir of model.')
    parser.add_argument('--test_list', type=str, default='./Landmark_Annotation', help='key image embeddings saving dictionary.')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    parser.add_argument('--backbone', type=str, default='swinv2', help='testing backbone')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    if args.backbone == 'swinv1':
        model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.backbone == 'swinv2':
        model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                                    num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    
    tsne(device, model=model, model_path=args.model_path, save_name="./test_consistency/ACEv2_swinv2_large_",flip=0, args=args)
