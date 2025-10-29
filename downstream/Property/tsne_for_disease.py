# tsne for downstream disease

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
import shutil
import sys
# Directory with the text files
text_files_dir = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/data/data_split/RSNA/RSNAPneumonia_test.txt'
# Directory with the png files
images_dir = '/sda1/zhouziyu/ssl/dataset/RSNA/stage_2_train_images_png/'
# image with annotations
images_anno = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/image_landmarks/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)


import numpy as np


def tsne(model_path="./POC_R_T_L.pth",save_name="tsne_plot",flip=0):
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
    output_class1 = []
    output_class2 = []
    output_class3 = []
    class1 = 0 # rsna数据集有三个类别，统计每个类别图像的数量
    class2 = 0
    class3 = 0
    with open(text_files_dir, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # Open the file
            line = line.strip()
            file_name = line.split(' ')[0]
            label = line.split(' ')[1]
            # Read the image
            img = cv2.imread(os.path.join(images_dir, file_name))
            img = cv2.resize(img, (448,448))
            img = transform(img).unsqueeze(0).to(device)
            #print(patch.shape)
            with torch.no_grad():
                # Extract features using the model
                features = model.forward_features(img) # swin:[1,196,768] vit:[1,197,768]
                features = features.mean(dim=1)
                # feature_vectors.append(features.cpu().numpy())

            if label=='0':
                class1+=1
                output_class1.append(features.cpu().numpy())
            elif label=='1':
                class2+=1
                output_class2.append(features.cpu().numpy())
            elif label=='2':
                class3+=1
                output_class3.append(features.cpu().numpy())



    output = output_class1+output_class2+output_class3
    output = np.array(output)

    print(class1, class2, class3)
    print(output.shape)


    # The rest of the code remains the same

    # Convert the output list to a numpy array


    # Reshape the output to 2D (11000, 1024)
    
    output = output.reshape(-1, output.shape[-1])
    # output2 = output2.reshape(-1, output2.shape[-1])
    print(output.shape)
    # np.save(save_name+".npy", output)
    #print(output.shape)
    # Use t-SNE to reduce dimensionality to 2
    tsne = TSNE(n_components=2,learning_rate=500,perplexity=40, random_state=42)
    output_tsne = tsne.fit_transform(output)

    x_min, x_max = np.min(output_tsne, 0), np.max(output_tsne, 0)
    output_tsne = output_tsne / (x_max - x_min)
    print(output_tsne.shape)


    # Colors for the scatter plot
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta', 'tomato', 'darkkhaki', 'silver']
    colors = ['blue', 'red', 'green']


    plt.scatter(output_tsne[:class1, 0], output_tsne[:class1, 1], color=colors[0], alpha=0.5)
    plt.scatter(output_tsne[class1:class1+class2, 0], output_tsne[class1:class1+class2, 1], color=colors[1], alpha=0.5)
    plt.scatter(output_tsne[class1+class2:class1+class2+class3, 0], output_tsne[class1+class2:class1+class2+class3, 1], color=colors[2], alpha=0.5)
    
    plt.savefig(save_name+".png")
    plt.close() 

    # for i in range(len(colors)):
    #     if i != len(colors):
    #         plt.scatter(output_tsne2[i*a:(i+1)*a, 0], output_tsne2[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)

    # plt.savefig(save_name+"_448.png")
    # plt.close() 
tsne(model_path='/sda1/zhouziyu/ssl/downstream_checkpoints/RSNAPneumonia/contrast_12n_global_inequalswin_base_linearprob_448_1/best.pth',save_name="./images/disease_rsna_linearprob",flip=0)

