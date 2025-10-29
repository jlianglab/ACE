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
from transformers import AutoModel
# Directory with the text files
text_files_dir = './Landmark_Annotation'
# Directory with the png files
images_dir = '/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/'
# image with annotations
images_anno = './images/image_landmarks/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)

# model = convnext.__dict__['convnext_base']()
# model = AutoModel.from_pretrained('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model

import numpy as np

def crop_and_pad(image, center, size=(512, 512)):
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
        checkpoint = checkpoint['student']
        # checkpoint = checkpoint['state_dict']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    
    if 'head.weight' in checkpoint_model:
        del checkpoint_model['head.weight']
    if 'head.bias' in checkpoint_model:
        del checkpoint_model['head.bias']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    model.cuda()
    output = []
    output2 = []
    # flip = 0
    # positions = [2,10,18,34,42,50,21]# [2,10,18,34,42,50,21] [21,2,10,18] [21,34,42,50]
    positions = [2,34,21,24,10,44,54,53,30]
    # positions = [10]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [position - 1 for position in positions]
    # Iterate over each file in the directory
    for pos in selected_positions:
        # if pos==99:
        #     flip = 1
        #     pos = 2
        # else:
        #     flip = 0
        filelist = []
        for file_name in os.listdir(text_files_dir):
            # Open the file
            # try:
                
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

                    selected_coord = coords[pos]

                    # Read the image
                    img = cv2.imread(os.path.join(images_dir, image_name))
                    filelist.append(image_name)

                    # For each coordinate, get the 224x224 patch around it
                    feature_vectors = []
                    x,y = selected_coord

                    # patch = img[ max(0, y -112):min(img.shape[1], y + 112), # crop 224 from 1024
                    #             max(0, x - 112):min(img.shape[0], x + 112) ]
                    patch = crop_and_pad(img, selected_coord)
                    patch = cv2.resize(patch, (512, 512), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)
                    if flip==1:
                        print("fliping")
                        patch = ImageOps.mirror(patch)
                        print("fliping finished")


                    # # Create the flipped version of the patch
                    # flipped_patch = ImageOps.mirror(patch)

                    # # Transform the original patch
                    # patch = transform(patch).unsqueeze(0).to(device)

                    # # Transform the flipped patch
                    # flipped_patch = transform(flipped_patch).unsqueeze(0).to(device)

                    patch = transform(patch).unsqueeze(0).to(device)
                    #print(patch.shape)
                    with torch.no_grad():
                        # Extract features using the model
                        _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                        # features = model(patch) # hugging face
                        # features = features.last_hidden_state[:,1:]
                        # print(features.shape)
                        f1 = features[:,119] # 90 for swinv1 backbone, 119 for swinv2 and 91 for vit backbone
                        # f2 = features[:,92]
                        # f3 = features[:,105]
                        # f4 = features[:,106]

                        # features = features.mean(dim=1) #[:,1:]
                        # features = (f1+f2+f3+f4)/4
                        features = f1
                        # feature_vectors.append(features.cpu().numpy())

                    # Concatenate these feature vectors
                    output.append(features.cpu().numpy())
                    # output.append(f1.cpu().numpy())
                    # output.append(f2.cpu().numpy())
                    # output.append(f3.cpu().numpy())
                    # output.append(f4.cpu().numpy())

                    # patch2 = img[ max(0, y -224):min(img.shape[1], y + 224), # crop 448 from 1024
                    #             max(0, x - 224):min(img.shape[0], x + 224) ]
                    # patch2 = cv2.resize(patch2, (448, 448), interpolation=cv2.INTER_CUBIC)
                    # patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
                    # patch2 = transform(patch2).unsqueeze(0).to(device)
                    # #print(patch.shape)
                    # with torch.no_grad():
                    #     features = model.forward_features(patch2) #_features
                    #     features = features.mean(dim=1) #[:,1:]
                    # output2.append(features.cpu().numpy())
            # except:
            #     print(file_name)
            #     continue
    # sys.exit(1)
    # print(len(output)//11)
    a=len(output)//len(positions)
    # a=len(output)//7
    # Convert the output list to a numpy array
    output = np.array(output)
    # output2 = np.array(output2)


    # The rest of the code remains the same

    # Convert the output list to a numpy array


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

    np.save(save_name+".npy", output_tsne)
    # output_tsne2 = tsne.fit_transform(output2)
    # x_min, x_max = np.min(output_tsne2, 0), np.max(output_tsne2, 0)
    # output_tsne2 = output_tsne2 / (x_max - x_min)

    if len(positions)==1:
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(output_tsne[0:a])
        print(labels)
        print(labels.shape)

        for i in range(len(labels)):
            if labels[i]==0:
                shutil.copy(os.path.join(images_anno, filelist[i]), os.path.join('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/red_patchembd_1',filelist[i]))
            elif labels[i]==1:
                shutil.copy(os.path.join(images_anno, filelist[i]), os.path.join('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/red_patchembd_2',filelist[i]))
            # elif labels[i]==2:
            #     shutil.copy(os.path.join(images_anno, filelist[i]), os.path.join('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/red_3',filelist[i]))


    # Colors for the scatter plot
    if positions == [21,2,10,18]:
        colors = ['brown','blue', 'red', 'green']
    elif positions == [21,34,42,50]:
        colors = ['brown','yellow', 'orange', 'purple']
    elif positions == [2,34,21,24,10,44,54,53,30]:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
    # colors = ['red']
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # Create a scatter plot
    
    colors = [
    '#87CEFA',  # light blue
    '#FF7F7F',  # light red
    '#90EE90',  # light green
    "#D1D171",  # light yellow
    '#FFD580',  # light orange
    '#D8BFD8',  # light purple
    '#FFB6C1',  # light magenta/pinkish
    "#A7A5A6",  # light pink
    "#BAF0F0",  # light cyan
]

    if output_tsne.shape[0]/a==28: # plot 4 tsne at the same time
        for i in range(28):
            if i %4==0:
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[int(i/4)], alpha=0.5)
        plt.savefig(save_name+"_1.png")
        plt.close() 
        for i in range(28):
            if i%4==1:
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[int(i/4)], alpha=0.5)
        plt.savefig(save_name+"_2.png")
        plt.close() 
        for i in range(28):
            if i%4==2:
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[int(i/4)], alpha=0.5)
        plt.savefig(save_name+"_3.png")
        plt.close() 
        for i in range(28):
            if i%4==3:
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[int(i/4)], alpha=0.5)
        plt.savefig(save_name+"_4.png")
        plt.close() 
    else:
        for i in range(len(positions)):
            if i != len(colors):
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
                # plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], c=labels, alpha=0.5)
                # Use triangle marker for i = 7

            else:
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5, marker='^')
        
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
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/comp_decomp/pretrained_weight/hierar_comp_decomp/checkpoint0100.pth',save_name="./images/tsne_plot_hierar_comp_decomp",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/consistency/pretrained_weight/global_local/checkpoint0100.pth',save_name="./images/tsne_plot_globloc_cosis",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/extropolation/pretrained_weight/extrapolation/checkpoint0100.pth',save_name="./images/tsne_plot_extrap_swin",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/extropolation/pretrained_weight/extrapolation/checkpoint0100.pth',save_name="./images/tsne_plot_rad-dino_swin",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/patch_shuffling/pretrained_weight/global_local_consistency/checkpoint0100.pth',save_name="./images/tsne_plot_shuffle_global_local_consistency",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/adam/Adam-v2_convnext_base.pth',save_name="./images/tsne_plot_adamv2",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint0050.pth',save_name="./images/from_mim_sslgenesis_50epc",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint.pth',save_name="./images/from_mim_extrap_shuffle_compdecomp",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/fromIN_extrap_shuffle_compdecomp/checkpoint.pth',save_name="./images/fromIN_extrap_shuffle_compdecomp",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/sslgenesis_ablation/extrapolation/pretrained_weight/random_mask/checkpoint0100.pth',save_name="./images/random_mask_100epc",flip=0)
tsne(model_path='/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',save_name="./images/ACEv2_swinv2_fromIN_changecolor",flip=0)


