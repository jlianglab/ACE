# compute similarity between the embedding of bbox and the whole image

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
import pandas as pd


#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    """
    # print(vec1.shape)
    # print(vec2.shape)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def get_disease_box(csv_file, disease='Atelectasis'):
    data = pd.read_csv(csv_file)

    # 筛选出某一疾病的所有行
    filter_data = data[data['Finding Label']==disease]

    imagebox_dict = {}
    for idx, row in filter_data.iterrows():
        bbox = (row['Bbox [x'], row['y'], row['w'], row['h]'])
        imagebox_dict[row['Image Index']] = bbox

    return imagebox_dict


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




def similarity(model_path, images_dir, imagebox_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the Swin Transformer model
    model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32), num_classes=3)

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
    # del checkpoint_model['head.weight']
    # del checkpoint_model['head.bias']
    # with open('statedict.txt', 'w') as f:
    #         for i in range(len(list(checkpoint_model.keys()))):
    #             f.writelines(list(checkpoint_model.keys())[i]+'\n')
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # for key in checkpoint_model.keys():
    #     #print(key)
    #     if key in model.state_dict().keys():
    #         try:
    #             model.state_dict()[key].copy_(checkpoint_model[key])
    #             print("Copying {} <---- {}".format(key, key))
    #         except:
    #             pass
            
    #     else:
    #         pass
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    normalize = transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    model.cuda()
    output_box = []
    output_whole = []

    for file_name, box in imagebox_dict.items(): # box:xywh
        box = tuple(int(float(value)) for value in box)
        whole_img = cv2.imread(os.path.join(images_dir, file_name))

        box_img = whole_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        box_img = cv2.resize(box_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        box_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
        box_img = Image.fromarray(box_img)
        box_img = transform(box_img).unsqueeze(0).to(device)

        whole_img = cv2.resize(whole_img, (448, 448), interpolation=cv2.INTER_CUBIC)
        whole_img = cv2.cvtColor(whole_img, cv2.COLOR_BGR2RGB)
        whole_img = Image.fromarray(whole_img)
        whole_img = transform(whole_img).unsqueeze(0).to(device)
        #print(patch.shape)
        with torch.no_grad():
            # Extract features using the model
            box_features = model.forward_features(box_img) # swin:[1,196,1024] vit:[1,197,768]
            box_features = torch.mean(box_features, dim=1)
            box_features = box_features.squeeze()
            whole_features = model.forward_features(whole_img)
            whole_features = torch.mean(whole_features, dim=1)
            whole_features = whole_features.squeeze()


        output_box.append(box_features.cpu().numpy())
        output_whole.append(whole_features.cpu().numpy())

    cosine_similarities = [cosine_similarity(vec1, vec2) for vec1, vec2 in zip(output_box, output_whole)]

    print(cosine_similarities)
    print(sum(cosine_similarities)/len(cosine_similarities))



#'/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth'
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth',save_name="./images/tsne_plot_vit",flip=0)





if __name__=="__main__":
    # Directory with the png files
    images_dir = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/'
    # image with annotations
    images_bbox = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/bbox/BBox_List_2017.csv'

    disease = 'Nodule'
    imagebox_dict = get_disease_box(images_bbox, disease)
    # print(imagebox_dict)

    model_path = '/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal.pth'
    # model_path = '/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth'
    # model_path = '/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/byol/checkpoint0300byol.pth'
    # model_path = '/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis/last.pth'
    similarity(model_path, images_dir, imagebox_dict)

