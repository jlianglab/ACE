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
import json
import ipdb
# Directory with the text files
text_files_dir = './Landmark_Annotation'
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    model.cuda()
    output = []
    output2 = []
    colors = plt.cm.get_cmap('tab20', 20)
    bgr_colors = [tuple(list(map(int, np.array(colors(i)[:3]) * 255))[::-1]) for i in range(20)]


    img_dir = '/sda1/zhouziyu/ssl/dataset/VinDr-RibCXR_Dataset'
    with open('/sda1/zhouziyu/ssl/dataset/VinDr-RibCXR_Dataset/Annotations/train/Vindr_RibCXR_train_mask.json', 'r') as file:
        data = json.load(file)
        image_file = data['img']

    img0 = cv2.imread('/sda1/zhouziyu/ssl/dataset/VinDr-RibCXR_Dataset/data/train/img/VinDr_RibCXR_train_000.png')
    h0, w0, _ = img0.shape
    img0 = cv2.resize(img0, (w0//2,h0//2))
    for i in range(20):
        if i%2==1:
            if i < 10:
                ribname = 'R'+str(i+1)
            else:
                ribname = 'L'+str(i-9)
            
            rib_data = data[ribname]

            for k,v in rib_data.items():
                image_path = os.path.join(img_dir, image_file[k])
                img = cv2.imread(image_path)
                h0, w0, _ = img.shape

                

                position = (int(v[0]['x']/2), int(v[0]['y']/2)) # 选取mask的第一个点
                img = cv2.resize(img, (w0//2,h0//2))
                # ipdb.set_trace()
                patch = crop_and_pad(img, position)
                patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = Image.fromarray(patch)
                patch = transform(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    # Extract features using the model
                    features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                    # print(features.shape)
                    f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                    # f2 = features[:,92]
                    # f3 = features[:,105]
                    # f4 = features[:,106]

                    # features = features.mean(dim=1) #[:,1:]
                    # features = (f1+f2+f3+f4)/4
                    features = f1
                    # feature_vectors.append(features.cpu().numpy())

                    # Concatenate these feature vectors
                    output.append(features.cpu().numpy())

                # draw landmarks
                if k=='0':
                    cv2.circle(img0, (int(v[0]['x']/2), int(v[0]['y']/2)), radius=15, color=bgr_colors[i], thickness=-1)  # 使用颜色绘制圆点
                if k=='0' and i==19:
                    print("Writing landmark image...")
                    success = cv2.imwrite('./images/vindr_ribs_20landmarks/img0.png', img0)
                    print(success)



    

    # # flip = 0
    # # positions = [2,10,18,34,42,50,21]# [2,10,18,34,42,50,21] [21,2,10,18] [21,34,42,50]
    # positions = [2,34,21,24,10,44,54,53,30]
    # # positions = [10]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    # selected_positions = [position - 1 for position in positions]
    # # Iterate over each file in the directory
    # for pos in selected_positions:
    #     # if pos==99:
    #     #     flip = 1
    #     #     pos = 2
    #     # else:
    #     #     flip = 0
    #     filelist = []
    #     for file_name in os.listdir(text_files_dir):
    #         # Open the file
    #         try:
                
    #             with open(os.path.join(text_files_dir, file_name), 'r') as f:
                    
    #                 # Read the content
    #                 content = f.read().strip()
    #                 # Split the content to get image name and coordinates
    #                 image_name, *coords = content.split('#')
    #                 image_name = image_name.split('-')[0] + '.png'
    #                 # Parse the coordinates
    #                 coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
    #                 #print(len(coords))
    #                 if len(coords)<54:
    #                     continue

    #                 # Randomly select 11 coordinates

    #                 selected_coord = coords[pos]

    #                 # Read the image
    #                 img = cv2.imread(os.path.join(images_dir, image_name))
    #                 filelist.append(image_name)

    #                 # For each coordinate, get the 224x224 patch around it
    #                 feature_vectors = []
    #                 x,y = selected_coord

    #                 # patch = img[ max(0, y -112):min(img.shape[1], y + 112), # crop 224 from 1024
    #                 #             max(0, x - 112):min(img.shape[0], x + 112) ]
    #                 patch = crop_and_pad(img, selected_coord)
    #                 patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
    #                 patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    #                 patch = Image.fromarray(patch)
    #                 if flip==1:
    #                     print("fliping")
    #                     patch = ImageOps.mirror(patch)
    #                     print("fliping finished")


    #                 # # Create the flipped version of the patch
    #                 # flipped_patch = ImageOps.mirror(patch)

    #                 # # Transform the original patch
    #                 # patch = transform(patch).unsqueeze(0).to(device)

    #                 # # Transform the flipped patch
    #                 # flipped_patch = transform(flipped_patch).unsqueeze(0).to(device)

    #                 patch = transform(patch).unsqueeze(0).to(device)
    #                 #print(patch.shape)
    #                 with torch.no_grad():
    #                     # Extract features using the model
    #                     features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #                     # print(features.shape)
    #                     f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
    #                     # f2 = features[:,92]
    #                     # f3 = features[:,105]
    #                     # f4 = features[:,106]

    #                     # features = features.mean(dim=1) #[:,1:]
    #                     # features = (f1+f2+f3+f4)/4
    #                     features = f1
    #                     # feature_vectors.append(features.cpu().numpy())

    #                 # Concatenate these feature vectors
    #                 output.append(features.cpu().numpy())
    #                 # output.append(f1.cpu().numpy())
    #                 # output.append(f2.cpu().numpy())
    #                 # output.append(f3.cpu().numpy())
    #                 # output.append(f4.cpu().numpy())

    #                 # patch2 = img[ max(0, y -224):min(img.shape[1], y + 224), # crop 448 from 1024
    #                 #             max(0, x - 224):min(img.shape[0], x + 224) ]
    #                 # patch2 = cv2.resize(patch2, (448, 448), interpolation=cv2.INTER_CUBIC)
    #                 # patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
    #                 # patch2 = transform(patch2).unsqueeze(0).to(device)
    #                 # #print(patch.shape)
    #                 # with torch.no_grad():
    #                 #     features = model.forward_features(patch2) #_features
    #                 #     features = features.mean(dim=1) #[:,1:]
    #                 # output2.append(features.cpu().numpy())
    #         except:
    #             print(file_name)
    #             continue

    # a=len(output)//20
    a=len(output)//10
    # a=len(output)//7
    # Convert the output list to a numpy array
    output = np.array(output)



    # Reshape the output to 2D (11000, 1024)
    
    output = output.reshape(-1, output.shape[-1])
    # output2 = output2.reshape(-1, output2.shape[-1])
    print(output.shape)
    # np.save(save_name+".npy", output)
    #print(output.shape)
    # Use t-SNE to reduce dimensionality to 2
    tsne = TSNE(n_components=2,learning_rate=500,perplexity=50)
    output_tsne = tsne.fit_transform(output)

    x_min, x_max = np.min(output_tsne, 0), np.max(output_tsne, 0)
    output_tsne = output_tsne / (x_max - x_min)
    print(output_tsne.shape)

    # Colors for the scatter plot

    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink','magenta','pink', 'cyan']
    
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
    # colors = ['red']
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # Create a scatter plot

    # for i in range(20):
    for i in range(20):
        if i%2==1:
            
        # plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors(i), alpha=0.5, marker='^')
            plt.scatter(output_tsne[i//2*a:(i//2+1)*a, 0], output_tsne[i//2*a:(i//2+1)*a, 1], color=colors(i), alpha=0.5)
    
    plt.savefig(save_name+".png")
    plt.close() 

    # for i in range(len(colors)):
    #     if i != len(colors):
    #         plt.scatter(output_tsne2[i*a:(i+1)*a, 0], output_tsne2[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)

    # plt.savefig(save_name+"_448.png")
    # plt.close() 
#'/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth'
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth',save_name="./images/tsne_plot_vit",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dino_vit_checkpoint0300.pth',save_name="./images/dino_tsne_vit",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal.pth',save_name="./images/tsne_plot_ace_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/simmim/ckpt_epoch_100.pth',save_name="./images/tsne_plot_simmim_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth',save_name="./images/tsne_plot_dino_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis/last.pth',save_name="./images/tsne_plot_PEAC_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/popar_ablations/POPAR_Swin_448.pth',save_name="./images/tsne_plot_POPAR_swin",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dropPos_vit-b32_448/droppos.pth',save_name="./images/tsne_plot_droppos_vit",flip=0)
tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/sslgenesis_peac/checkpoint0150.pth',save_name="./images/tesne_vindr_10ribs_sslgenesis_150epc",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/extrapolation/pretrained_weight/extrapolation_MIM/checkpoint0100.pth',save_name="./images/tsne_plot_extrap_MIM_swin",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_mlp_overlapglobal/checkpoint0050.pth',save_name="./images/from_imagenet_matrixcompdecomp_mlp_overlapglobal_50epc_9cluster",flip=0)

