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
from timm.models.swin_transformer import SwinTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image, ImageOps
import torch.nn as nn
from sklearn.cluster import KMeans
import sys
import shutil
# Directory with the text files
text_files_dir = './Landmark_Annotation'
# text_files_dir = './Landmark_test'
# Directory with the png files
images_dir = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/'
# image with annotations
images_anno = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/image_landmarks/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)


import numpy as np


def plot_line_chart(data_list, save_name='swin_mirror', backbone_name='Swin-B'):
    """
    Plots a line chart of the given list of data.
    Assumes the list represents the y-values and uses the index as the x-values.

    :param data_list: List of data points (y-values)
    """
    x_values = range(len(data_list))  # x-values are the indices of the data list
    y_values = data_list  # y-values are the data points

    plt.plot(x_values, y_values, marker='o')  # 'o' marker to show points on the line
    plt.title(backbone_name)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(save_name+".png")
    plt.close() 

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    :param vec1: First vector
    :param vec2: Second vector
    :return: Cosine similarity
    """
    vec1 = vec1[0]
    vec2 = vec2[0] # [1024]
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def crop_and_pad(image, center, size=(448, 448)):
    """
    Crops a square region of specified size from the image centered at the given point.
    If the region goes beyond the image boundaries, it's padded with zeros.
    
    :param image: NumPy array representing the image.
    :param center: Tuple (x, y) representing the center of the region to be cropped.
    :param size: Size of the square region to be cropped.
    :return: Cropped and padded image.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = size

    # Calculate crop boundaries
    start_x = max(center[0] - (crop_w // 2-16), 0)
    end_x = min(center[0] + crop_w // 2+16, w)
    start_y = max(center[1] - (crop_h // 2-16), 0)
    end_y = min(center[1] + (crop_h // 2+16), h)

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Calculate padding sizes
    pad_left = abs(min(center[0] - (crop_w // 2-16), 0))
    pad_right = crop_w - (end_x - start_x) - pad_left
    pad_top = abs(min(center[1] - (crop_w // 2-16), 0))
    pad_bottom = crop_h - (end_y - start_y) - pad_top

    # Pad the cropped image
    padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')

    return padded_image


def linear_interpolation(pointA, pointB, num_points=9):
    """
    Generates equally spaced interpolated points between two given points.

    :param pointA: Tuple (x, y) representing the first point.
    :param pointB: Tuple (x, y) representing the second point.
    :param num_points: Number of interpolated points to generate.
    :return: List of tuples representing the interpolated points (pointA+num_points+pointB).
    """
    x1, y1 = pointA
    x2, y2 = pointB
    interpolated_points = []

    for i in range(num_points + 2):
        t = i / (num_points + 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        interpolated_points.append((int(x), int(y)))

    return interpolated_points


def tsne(model_path="./POC_R_T_L.pth",interpolation=9):
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
    except:
        checkpoint = checkpoint['model']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    for key in checkpoint_model.keys():
        #print(key)
        if key in model.state_dict().keys():
            try:
                model.state_dict()[key].copy_(checkpoint_model[key])
            except:
                pass
            print("Copying {} <---- {}".format(key, key))
        else:
            pass
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    normalize = transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    model.cuda()
    output = []
    output2 = []
    center = [10]
    y_axis = [2,4,6,8,10,12,14,16,18,20]
    x_axis = [42]
    center = [a-1 for a in center]
    y_axis = [a-1 for a in y_axis]
    x_axis = [a-1 for a in x_axis]
    positions = [10,42]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [position - 1 for position in positions]
    # Iterate over each file in the directory

    img_num = 0
    y_similarity_total = []
    for file_name in os.listdir(text_files_dir):
        y_features = []
        try:
            
            for pos in y_axis:
            # Open the file
            
                
                with open(os.path.join(text_files_dir, file_name), 'r') as f:
                    
                    # Read the content
                    content = f.read().strip()
                    # Split the content to get image name and coordinates
                    image_name, *coords = content.split('#')
                    image_name = image_name.split('-')[0] + '.png'
                    # Parse the coordinates
                    coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
                    #print(len(coords))
                    if len(coords)<54:
                        continue

                    # Randomly select 11 coordinates
                    x_begin_coord = coords[center[0]]
                    x_end_coord = coords[x_axis[0]]
                    selected_coord = coords[pos]

                    # Read the image
                    img = cv2.imread(os.path.join(images_dir, image_name))

                    # patch = img[ max(0, y -112):min(img.shape[1], y + 112), # crop 224 from 1024
                    #             max(0, x - 112):min(img.shape[0], x + 112) ]
                    patch = crop_and_pad(img, selected_coord)
                    patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)

                    patch = transform(patch).unsqueeze(0).to(device)
                    print(patch.shape)
                    with torch.no_grad():
                        # Extract features using the model
                        features = model.forward_features(patch) # [1,197,768]
                        f1 = features[:,91] # 90 for swin backbone and 91 for vit backbone
            
                        features = f1
                        # feature_vectors.append(features.cpu().numpy())

                    # Concatenate these feature vectors
                    y_features.append(features.cpu().numpy())

            center_feature = y_features[4]
            y_similarity = []
            for i in y_features:
                y_similarity.append(cosine_similarity(center_feature, i))
            img_num+=1

        except Exception as e:
            print(e)
            print(file_name)
            continue
        if not y_similarity_total:
            y_similarity_total = y_similarity
        else:
            y_similarity_total = [x + y for x, y in zip(y_similarity_total, y_similarity)]
    print(img_num)
    # sys.exit(1)
    print(y_similarity_total)
    
    y_similarity_total = [x / img_num for x in y_similarity_total]
    plot_line_chart(y_similarity_total, './images/swin_mirror_y', 'Swin-B')

    
    x_similarity_total = []
    img_num = 0
    for file_name in os.listdir(text_files_dir):
        x_features = []
        try:
            
            with open(os.path.join(text_files_dir, file_name), 'r') as f:
                    
                    # Read the content
                    content = f.read().strip()
                    # Split the content to get image name and coordinates
                    image_name, *coords = content.split('#')
                    image_name = image_name.split('-')[0] + '.png'
                    # Parse the coordinates
                    coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
                    #print(len(coords))
                    if len(coords)<54:
                        continue

                    # Randomly select 11 coordinates
                    x_begin_coord = coords[center[0]]
                    x_end_coord = coords[x_axis[0]]
                    x_coords = linear_interpolation(x_begin_coord, x_end_coord, interpolation)
                    img = cv2.imread(os.path.join(images_dir, image_name))
            for selected_coord in x_coords:
                patch = crop_and_pad(img, selected_coord)
                patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = Image.fromarray(patch)

                patch = transform(patch).unsqueeze(0).to(device)
                #print(patch.shape)
                with torch.no_grad():
                    # Extract features using the model
                    features = model.forward_features(patch) # [1,197,768]
                    f1 = features[:,91]
                    features = f1
                    # feature_vectors.append(features.cpu().numpy())

                # Concatenate these feature vectors
                x_features.append(features.cpu().numpy())

            center_feature = x_features[0]
            x_similarity = []
            for i in x_features:
                x_similarity.append(cosine_similarity(center_feature, i))
            img_num+=1
        except:
            print('pass')
            continue
        if not x_similarity_total:
            x_similarity_total = x_similarity
        else:
            x_similarity_total = [x + y for x, y in zip(x_similarity_total, x_similarity)]
    print(x_similarity_total)
    print(img_num)
    x_similarity_total = [x / img_num for x in x_similarity_total]

    # x_similarity_total = [969.0, 779.60535, 665.6629, 579.6754, 526.17633, 502.11304, 497.54477, 497.83426, 508.24692, 544.79425, 616.66437]
    # divisor = 987
    # x_similarity_total = [x / divisor for x in x_similarity_total]

    plot_line_chart(x_similarity_total, './images/swin_mirror_x', 'Swin-B')
    






    # for i in range(len(colors)):
    #     if i != len(colors):
    #         plt.scatter(output_tsne2[i*a:(i+1)*a, 0], output_tsne2[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)

    # plt.savefig(save_name+"_448.png")
    # plt.close() 
#'/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth'
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth')
tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal.pth')

