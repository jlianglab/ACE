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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
# model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)

# model = convnext.__dict__['convnext_base']()
# model = AutoModel.from_pretrained('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model

import numpy as np

class GlobalFeatureMLP(nn.Module):
    def __init__(self, num_tokens=196, embed_dim=1024):
        super(GlobalFeatureMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens * embed_dim, embed_dim),  # Flatten (B, 196, 1024) -> (B, 196*1024) -> (B, 1024)
            nn.ReLU()  # Non-linearity
        )

    def forward(self, x):
        B, N, C = x.shape  # (B, 196, 1024)
        x = x.view(B, N * C)  # Flatten to (B, 196*1024)
        x = self.mlp(x)  # MLP to (B, 1024)
        return x

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.global_head = GlobalFeatureMLP()


    def forward(self, x):
        # convert to list
        _, features = self.backbone.forward_features(x)
        output = self.global_head(features)

        return output


def crop_and_resize(image, center, size=(448, 448), aspect_ratios=[1, 3/4, 4/3]):
    h, w, _ = image.shape
    crops = []
    
    for i in range(len(aspect_ratios)):
        ratio = aspect_ratios[i]
        if ratio == 1:  # 以中心点裁剪一个正方形
            half_size = size[0] // 2
            x1 = max(center[0] - half_size, 0)
            y1 = max(center[1] - half_size, 0)
            x2 = min(center[0] + half_size, w)
            y2 = min(center[1] + half_size, h)
        else:  # 裁剪长宽比为3/4或4/3的矩形
            if ratio > 1:  # 长宽比4/3
                new_width = int(size[0] * ratio / np.sqrt(1 + ratio**2))
                new_height = int(new_width / ratio)
            else:  # 长宽比3/4
                new_height = int(size[0] * np.sqrt(1 + 1/ratio**2))
                new_width = int(new_height * ratio)
            
            half_width = new_width // 2
            half_height = new_height // 2
            x1 = max(center[0] - half_width, 0)
            y1 = max(center[1] - half_height, 0)
            x2 = min(center[0] + half_width, w)
            y2 = min(center[1] + half_height, h)
        
        crop = image[y1:y2, x1:x2]
        # crop_resized = cv2.resize(crop, (448, 448))
        crop_resized = cv2.resize(crop, (512, 512)) # swinv2
        crops.append(crop_resized)
    
    return crops[0], crops[1], crops[2]




def tsne(model_path="./POC_R_T_L.pth",save_name="tsne_plot",flip=0):
    # encoder = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        #   num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    encoder = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                                        num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    # model = MultiCropWrapper(encoder,)
    model = encoder
    
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['teacher']
    except:
        checkpoint = checkpoint
        # checkpoint = checkpoint['model']
        # checkpoint = checkpoint['student']
        # checkpoint = checkpoint['state_dict']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("vit.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", "base_model."): v for k, v in checkpoint_model.items()}
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
    model.to(device)
    output1 = []
    output2 = []
    output3 = []
    # flip = 0
    # positions = [2,10,18,34,42,50,21]# [2,10,18,34,42,50,21] [21,2,10,18] [21,34,42,50]
    # positions = [2,34,21,24,10,44,54,53,30]
    # positions = [2,45,54,5]
    positions = [2,45,54]
    # positions = [10]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [position - 1 for position in positions]
    # Iterate over each file in the directory
    for pos in selected_positions:

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
                    # patch1, patch2, patch3 = crop_and_resize(img, selected_coord, (448, 448))
                    # patch1, patch2, patch3 = crop_and_resize(img, selected_coord, (96, 96)) # swinv1
                    patch1, patch2, patch3 = crop_and_resize(img, selected_coord, (512, 512)) # swinv2
                    # patch = crop_and_pad(img, selected_coord, (96, 96))
                    # patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                    patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB)
                    patch1 = Image.fromarray(patch1)
                    patch1 = transform(patch1).unsqueeze(0).to(device)
                    
                    patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
                    patch2 = Image.fromarray(patch2)
                    patch2 = transform(patch2).unsqueeze(0).to(device)
                    
                    patch3 = cv2.cvtColor(patch3, cv2.COLOR_BGR2RGB)
                    patch3 = Image.fromarray(patch3)
                    patch3 = transform(patch3).unsqueeze(0).to(device)
                    #print(patch.shape)
                    with torch.no_grad():
                        # Extract features using the model
                        _, features1 = model.forward_features(patch1)
                        _, features2 = model.forward_features(patch2)
                        _, features3 = model.forward_features(patch3)
                        # f1 = features1[:, 90] # swinv1
                        # f2 = features2[:, 90]
                        # f3 = features3[:, 90]
                        f1 = features1[:,119] # swinv2
                        f2 = features2[:,119]
                        f3 = features3[:,119]


                    # Concatenate these feature vectors
                    output1.append(f1.cpu().numpy())
                    output2.append(f2.cpu().numpy())
                    output3.append(f3.cpu().numpy())
                    



    

    a=len(output1)//(len(positions))

    output = output1+output2+output3

    output = np.array(output)



    
    output = output.reshape(-1, output.shape[-1])

    print(output.shape)

    # Use t-SNE to reduce dimensionality to 2
    tsne = TSNE(n_components=2,learning_rate=500,perplexity=50)
    output_tsne = tsne.fit_transform(output)

    x_min, x_max = np.min(output_tsne, 0), np.max(output_tsne, 0)
    output_tsne = output_tsne / (x_max - x_min)
    print(output_tsne.shape)

    # np.save(save_name+".npy", output_tsne)
    # if os.path.exists(save_name+".npy"):
    #     output_tsne = np.load(save_name+".npy")
    # a=output_tsne.shape[0]//(3*len(positions))
    total_l = output_tsne.shape[0]//3
        





    # Colors for the scatter plot
    if positions == [21,2,10,18]:
        colors = ['brown','blue', 'red', 'green']
    elif positions == [21,34,42,50]:
        colors = ['brown','yellow', 'orange', 'purple']
    elif positions == [2,34,21,24,10,44,54,53,30]:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    elif positions == [2,45,54,5]:
        colors = ['yellow', 'pink', 'green', 'blue']
    elif positions == [2,45,54]:
        # colors = ['yellow', 'pink', 'green']
        colors = ["#D1D171", 'pink', '#87CEFA']
        
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
    # colors = ['red']
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # Create a scatter plot

    if output_tsne.shape[0]/a==len(positions)*2: # plot 2 tsne at the same time
        for i in range(len(positions)):
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
        plt.savefig(save_name+"96.png")
        plt.close() 
        for i in range(len(positions),2*len(positions)):
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[int(i-len(positions))], alpha=0.5)
        plt.savefig(save_name+"448.png")
        plt.close() 
        
    else:
        for i in range(len(positions)):
            # if i != len(colors):
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
                plt.scatter(output_tsne[total_l+i*a:total_l+(i+1)*a, 0], output_tsne[total_l+i*a:total_l+(i+1)*a, 1], color=colors[i], alpha=0.5, marker='^')
                plt.scatter(output_tsne[total_l*2+i*a:total_l*2+(i+1)*a, 0], output_tsne[total_l*2+i*a:total_l*2+(i+1)*a, 1], color=colors[i], alpha=0.5, marker='s')
                

            # else:
            #     plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5, marker='^')
        
        plt.savefig(save_name+".png")
        plt.close() 

    # for i in range(len(colors)):
    #     if i != len(colors):
    #         plt.scatter(output_tsne2[i*a:(i+1)*a, 0], output_tsne2[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)

    # plt.savefig(save_name+"_448.png")
    # plt.close() 
# tsne(model_path='/mnt/nvme1n1/zhouziyu/sslgenesis_ablation/extrapolation/pretrained_weight/random_mask/checkpoint0100.pth',save_name="./images/random_mask_100epc",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/sslgenesis_ablation/symmetry/pretrained_weight/symmetry_local/checkpoint0050.pth',save_name="./images/symmetry_local",flip=0)
# tsne(model_path='/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',save_name="./images/ark5",flip=0)
# tsne(model_path='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/large_fromMIM_extrap_shuffle_consis_compdecomp/checkpoint0025.pth',save_name="./images/Lamps_large_fromMIM_crop",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/ACE_journal/uniqueness/pretrained_weight/uniqueness_multigranu_big_local_crop/checkpoint0158.pth',save_name="./images/Uniqueness_test_size96",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/ACE_journal/uniqueness/pretrained_weight/fromIN_uniqueness_multigranu_big_local_crop/checkpoint0150.pth',save_name="./images/Uniqueness_fromIN_150epc_size96",flip=0)
# tsne(model_path='/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',save_name="./images/ACEv2_swinv1_test_uniqueness_size448",flip=0)
tsne(model_path='/sda/zhouziyu/ssl/pretrained_model/ACE_v2/large_swinv2_fromIN_unique_multiscale_consis_compdecomp/checkpoint0020.pth',save_name="./images/ACEv2_swinv2_test_uniqueness_large_size512",flip=0)
# tsne(model_path='/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',save_name="./images/ACEv2_swinv2_test_uniqueness_nih_size512",flip=0)

