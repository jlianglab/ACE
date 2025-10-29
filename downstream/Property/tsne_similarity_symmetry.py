# test symmetry property by flipping features

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image, ImageOps
import torch.nn as nn
from sklearn.cluster import KMeans
import shutil
import sys
import models.convnext as convnext
import ipdb
# Directory with the text files
text_files_dir = './Landmark_Annotation'
# Directory with the png files
images_dir = '/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/'
# image with annotations
images_anno = './images/image_landmarks/'

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3)
model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
model = model.to(device)

model2 = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
model2 = model2.to(device)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)

# model = convnext.__dict__['convnext_base']()

import numpy as np

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



def tsne(model_path="./POC_R_T_L.pth",save_name="tsne_plot",flip=0):
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

    checkpoint2 = torch.load('/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/fromscratch_extrap_popar_compdecomp_consis/checkpoint0150.pth', map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint2['teacher']
    except:
        # checkpoint = checkpoint['model']
        checkpoint = checkpoint2['state_dict']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}

    msg = model2.load_state_dict(checkpoint_model, strict=False)
    print(msg)
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    # model.cuda()
    output = []
    output1 = []
    output2 = []
    # colors = plt.cm.get_cmap('tab20', 12)
    colors = ['blue', 'red', 'green', 'cyan']
    # colors = ['blue','yellow','cyan', 'red','magenta','orange', 'green','purple','brown']
    # bgr_colors = [tuple(list(map(int, np.array(colors(i)[:3]) * 255))[::-1]) for i in range(12)]

    # positions = [2,12,54]
    positions = [2,34] # left/right clavicle
    positions_sym = [34,44,53]

    selected_positions = [position - 1 for position in positions]
    selected_positions_sym = [position - 1 for position in positions_sym]

    pos = selected_positions[0]
    pos_sym = selected_positions_sym[0]




    # # right
    # for file_name in os.listdir(text_files_dir):
    #     with open(os.path.join(text_files_dir, file_name), 'r') as f:
            
    #         # Read the content
    #         content = f.read().strip()
    #         # Split the content to get image name and coordinates
    #         image_name, *coords = content.split('#')
    #         image_name = image_name.split('-')[0] + '.png'
    #         # Parse the coordinates
    #         coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
    #         #print(len(coords))
    #         if len(coords)<54:
    #             continue

    #         selected_coord = coords[pos_sym]

    #         # Read the image
    #         img = cv2.imread(os.path.join(images_dir, image_name))


    #         # patch = img[ max(0, y -112):min(img.shape[1], y + 112), # crop 224 from 1024
    #         #             max(0, x - 112):min(img.shape[0], x + 112) ]
    #         patch = crop_and_pad(img, selected_coord)
    #         patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
    #         patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    #         patch = Image.fromarray(patch)


    #         patch = transform(patch).unsqueeze(0).to(device)
    #         #print(patch.shape)
    #         with torch.no_grad():
    #             # Extract features using the model
    #             # ipdb.set_trace()
    #             # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #             _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #             # print(features.shape)
    #             f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

    #             features = f1
    #             # feature_vectors.append(features.cpu().numpy())

    #         # Concatenate these feature vectors
    #         output2.append(features.cpu().numpy())
    #         output.append(features.cpu().numpy()) 

    
    # left
    for pos in selected_positions:
        for file_name in os.listdir(text_files_dir):
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
                #print(patch.shape)
                with torch.no_grad():
                    # Extract features using the model
                    # ipdb.set_trace()
                    # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                    _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                    # print(features.shape)
                    f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

                    features = f1
                    # feature_vectors.append(features.cpu().numpy())

                # Concatenate these feature vectors
                output.append(features.cpu().numpy())


    # left feature flip
    for pos in selected_positions:
        for file_name in os.listdir(text_files_dir):
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
                #print(patch.shape)
                with torch.no_grad():
                    # Extract features using the model
                    # ipdb.set_trace()
                    # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                    _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                    # print(features.shape)
                    f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                    f1 = f1.view(1,32,32)
                    f1 = f1.flip(dims=[2])
                    f1 = f1.view(1,-1)
                    features = f1
                    # feature_vectors.append(features.cpu().numpy())

                # Concatenate these feature vectors
                output.append(features.cpu().numpy()) 



    # # left flip
    # for pos in selected_positions:
    #     for file_name in os.listdir(text_files_dir):
    #         with open(os.path.join(text_files_dir, file_name), 'r') as f:
                
    #             # Read the content
    #             content = f.read().strip()
    #             # Split the content to get image name and coordinates
    #             image_name, *coords = content.split('#')
    #             image_name = image_name.split('-')[0] + '.png'
    #             # Parse the coordinates
    #             coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
    #             #print(len(coords))
    #             if len(coords)<54:
    #                 continue

    #             selected_coord = coords[pos]

    #             # Read the image
    #             img = cv2.imread(os.path.join(images_dir, image_name))


    #             # patch = img[ max(0, y -112):min(img.shape[1], y + 112), # crop 224 from 1024
    #             #             max(0, x - 112):min(img.shape[0], x + 112) ]
    #             patch = crop_and_pad(img, selected_coord)
    #             # ipdb.set_trace()
    #             patch = np.flip(patch, axis=1)
    #             patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
    #             patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    #             patch = Image.fromarray(patch)


    #             patch = transform(patch).unsqueeze(0).to(device)
    #             #print(patch.shape)
    #             with torch.no_grad():
    #                 # Extract features using the model
    #                 # ipdb.set_trace()
    #                 # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #                 _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #                 # print(features.shape)
    #                 f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
    #                 
    #                 features = f1
    #                 # feature_vectors.append(features.cpu().numpy())

    #             # Concatenate these feature vectors
    #             output1.append(features.cpu().numpy()) # left flip feature
    #             output.append(features.cpu().numpy()) # left flip feature


    # compute similarity between left-right padding feature of left lung and right lung
    # similarties_list = []
    # for i in range(len(output1)):
    #     sim = cosine_similarity(output1[i][0], output2[i][0])
    #     print(sim)
    #     similarties_list.append(sim)
    # similarties_list = np.array(similarties_list)
    # np.save('./symmetry_similarity/SymmetryTrain_leftcropflip_right.npy', similarties_list)


    # tsne
    a=len(output)//4
    output = np.array(output)
    output = output.reshape(-1, output.shape[-1])
    # ipdb.set_trace()

    tsne = TSNE(n_components=2,learning_rate=500,perplexity=50)
    output_tsne = tsne.fit_transform(output)

    x_min, x_max = np.min(output_tsne, 0), np.max(output_tsne, 0)
    output_tsne = output_tsne / (x_max - x_min)
    print(output_tsne.shape)

    for i in range(4):
        print(colors[i])
        plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
    plt.savefig('./symmetry_similarity/SymmetryTrain_leftrightclavicle_embdflip.png')
    plt.close() 





#'/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth'
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth',save_name="./images/tsne_plot_vit",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dino_vit_checkpoint0300.pth',save_name="./images/dino_tsne_vit",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal.pth',save_name="./images/tsne_plot_ace_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/simmim/ckpt_epoch_100.pth',save_name="./images/tsne_plot_simmim_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth',save_name="./images/tsne_plot_dino_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis/last.pth',save_name="./images/tsne_plot_PEAC_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/popar_ablations/POPAR_Swin_448.pth',save_name="./images/tsne_plot_POPAR_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dropPos_vit-b32_448/droppos.pth',save_name="./images/tsne_plot_droppos_vit",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/comp_decomp/pretrained_weight/hierar_comp_decomp/checkpoint0100.pth',save_name="./images/tsne_plot_hierar_comp_decomp",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/consistency/pretrained_weight/global_local/checkpoint0100.pth',save_name="./images/tsne_plot_globloc_cosis",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/extropolation/pretrained_weight/extrapolation/checkpoint0100.pth',save_name="./images/symmetry_tsne_plot_extrap_swin",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/sslgenesis_peac/checkpoint0150.pth',save_name="./images/symmetry_8points_tsne_plot_sslgenesis_150epc",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/fromMIM_extrap_shuffle_compdecomp/checkpoint0075.pth',save_name="./images/symmetry_6points_from_simmim_extrap_shuffle_compdecomp_75epc",flip=0)
# tsne(model_path='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/fromscratch_extrap_popar_compdecomp_consis/checkpoint0150.pth')
tsne(model_path='/mnt/nvme1n1/zhouziyu/sslgenesis_ablation/symmetry/pretrained_weight/symmetry_global/checkpoint0050.pth')
