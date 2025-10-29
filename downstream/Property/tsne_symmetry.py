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
import models.convnext as convnext
from transformers import AutoModel
from models.resnet import resnet50
from models.eva_x import eva_x_base_patch16
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
import csv
import ipdb
# Directory with the text files
text_files_dir = './Landmark_Annotation'
# image with annotations
images_anno = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/images/image_landmarks/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3)
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
#                           num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
# model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
#                                 num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
# model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
#                         mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                         drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)

# model = convnext.__dict__['convnext_base']()

import numpy as np

def crop_and_pad(image, center, size=(448, 448), stride=32):
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
    start_x = int(max(center[0] - (crop_w // 2-stride//2), 0))
    end_x = int(min(center[0] + crop_w // 2+stride//2, w))
    start_y = int(max(center[1] - (crop_h // 2-stride//2), 0))
    end_y = int(min(center[1] + (crop_h // 2+stride//2), h))

    # ipdb.set_trace()

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Calculate padding sizes
    pad_left = int(abs(min(center[0] - (crop_w // 2-stride//2), 0)))
    pad_right = int(crop_w - (end_x - start_x) - pad_left)
    pad_top = int(abs(min(center[1] - (crop_w // 2-stride//2), 0)))
    pad_bottom = int(crop_h - (end_y - start_y) - pad_top)
    

    # Pad the cropped image
    padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')

    return padded_image




def tsne(model, model_path="./POC_R_T_L.pth",save_name="tsne_plot",args=None):
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    if args.pretrain_mode in ['adamv2']:
        checkpoint = checkpoint['teacher']
    elif args.pretrain_mode in ['EVA-X']:
        checkpoint = checkpoint['module']
    elif args.pretrain_mode in ['CheSS']:
        checkpoint = checkpoint['state_dict']
    else:
        checkpoint = checkpoint
    # except:
    #     checkpoint = checkpoint

    #checkpoint = checkpoint['student']
    # checkpoint_model = {k.replace("vit.", ""): v for k, v in checkpoint_model.items()}
    # checkpoint_model = {k.replace("backbone.", "base_model."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("encoder_q.", ""): v for k, v in checkpoint_model.items()}
    
    if 'head.weight' in checkpoint_model:
        del checkpoint_model['head.weight']
    if 'head.bias' in checkpoint_model:
        del checkpoint_model['head.bias']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    model.cuda()
    output = []
    output2 = []
    colors = plt.cm.get_cmap('tab20', 12)
    bgr_colors = [tuple(list(map(int, np.array(colors(i)[:3]) * 255))[::-1]) for i in range(12)]
    # flip = 0
    # positions = [2,10,18,34,42,50,21]# [2,10,18,34,42,50,21] [21,2,10,18] [21,34,42,50]
    # positions = [2,34,21,24,10,44,54,53,30]
    # positions = [2,34,10,42,8,40,12,44,14,46,16,48,21]
    positions = [2,34,21,8,40,12,44,16,48] # right clavicle,left clavicle,spinous process,right rib4,left rib4, right rib6, left rib6, right rib8, left rib8
    # positions = [2,34,10,42,16,48]
    # positions = [10,42,8,40,12,44,14,46,16,48]
    # positions = [10]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [position - 1 for position in positions]
    if args.add_syn:
        E_right_clavicle = []
        E_syn_left_clavicle = []
        E_left_clavicle = []
        E_spinous_process = []
        similarities = []
        
        file = open(args.save_file, mode='w', newline='', encoding='utf-8')
        landmark_name = ['ImageName', 'Similarity']
        writer = csv.writer(file)
        writer.writerow(landmark_name)  # 写入表头
        file.flush()
        
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
                    img = cv2.imread(os.path.join(args.image_dir, image_name))
                    filelist.append(image_name)

                    # For each coordinate, get the 224x224 patch around it
                    feature_vectors = []
                    x,y = selected_coord

                    if args.pretrain_mode in ['PEAC','ACE','Lamps', 'ACE-v2', 'ark']: # imgsize 448
                        shift = 32
                    elif args.pretrain_mode in ['LeADER', 'adamv2','EVA-X','RAD-DINO']: # imgsize 224
                        shift = 0
                    elif args.pretrain_mode in ['CheSS']:
                        shift = 28
                    patch = crop_and_pad(img, selected_coord, (args.img_size, args.img_size), shift)
                    patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)


                    patch = transform(patch).unsqueeze(0).to(device)
                    #print(patch.shape)
                    with torch.no_grad():
                        if args.embd_dim == 2048: # chess
                            patch = patch[:,0].unsqueeze(1)
                        if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS', 'ACE-v2', 'ark']:
                            _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
                        # ipdb.set_trace()
                        elif args.pretrain_mode in ['RAD-DINO']:
                            features = model(patch) # hugging face
                            features = features.last_hidden_state[:,1:] # hugging face
                        elif args.pretrain_mode in ['adamv2']:
                            features = model.extract_features(patch) # convnext

                        if args.backbone == 'swinv2':
                            f1 = features[:,119]
                        else:
                            if args.pretrain_mode in ['PEAC','ACE','Lamps','EVA-X', 'ACE-v2','ark']:
                                f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                            elif args.pretrain_mode in ['LeADER', 'adamv2']:
                                f1 = features[:,24]
                            elif args.pretrain_mode in ['RAD-DINO']:
                                f1 = features[:,684] # rad-dino has 1369(37*37) features
                            elif args.pretrain_mode in ['CheSS']:
                                f1 = features[:,119] # CheSS has 256(16*16) features

                        # features = features.mean(dim=1) #[:,1:]
                        # features = (f1+f2+f3+f4)/4
                        features = f1
                        
                        if args.add_syn:
                            if pos == 33: # right clavicle
                                E_right_clavicle.append(features.cpu().numpy())
                            elif pos == 20: # spinous process
                                E_spinous_process.append(features.cpu().numpy())
                            elif pos == 1: # left clavicle
                                E_left_clavicle.append(features.cpu().numpy())
                        # feature_vectors.append(features.cpu().numpy())

                    # Concatenate these feature vectors
                    output.append(features.cpu().numpy())

            # except:
            #     print(file_name)
            #     continue
    
    if args.add_syn:
        for i in range(len(E_right_clavicle)):
            E_syn = E_right_clavicle[i] + E_left_rib4[i] - E_right_rib4[i]
            E_syn_left_clavicle.append(E_syn)
            similarities.append(cosine_similarity(E_left_clavicle[i].reshape(1, -1), E_syn.reshape(1, -1))[0][0])
            writer.writerow([filelist[i], similarities[i]])
        file.flush()
        output.extend(E_syn_left_clavicle)
    # sys.exit(1)
    # ipdb.set_trace()
    # print(len(output)//11)
    a=len(output)//(len(positions))
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

    # output_tsne2 = tsne.fit_transform(output2)
    # x_min, x_max = np.min(output_tsne2, 0), np.max(output_tsne2, 0)
    # output_tsne2 = output_tsne2 / (x_max - x_min)

    # for i in range(12):
            
    #     plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors(i), alpha=0.5, marker='^')

    # plt.savefig(save_name+".png")
    # plt.close() 


    # Colors for the scatter plot
    # if positions == [21,2,10,18]:
    #     colors = ['brown','blue', 'red', 'green']
    # elif positions == [21,34,42,50]:
    #     colors = ['brown','yellow', 'orange', 'purple']
    # elif positions == [2,34,21,24,10,44,54,53,30]:
    #     colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    # else:
    #     colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink','magenta','cyan','slategray','sienna','skyblue','turquoise']
    
    
    colors = [ # positions = [2,34,21,24,10,44,54,53,30]
    '#87CEFA',  # light blue, right clavicle
    '#FF7F7F',  # light red, left clavicle
    '#90EE90',  # light green, spinous process
    '#FFD580',  # light orange, right rib4
    "#4682B4",  # steel blue, left rib4
    "#D1D171",  # light yellow, right rib6
    '#D8BFD8',  # light purple, left rib6
    '#FFB6C1',  # light magenta/pinkish, right rib8
    "#A7A5A6",  # light gray, left rib8
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
            # if i in [0,1,4,5,6,7,10,11,12]: # 8 points+spinous process
            # if i in [0,1,4,5,6,7,10,11]: # 8 points
            # if i in [0,1,4,5,10,11]: # 6 points
            # if i in [0,1,4,5]: # clavicel, rib4
            # if i in [4,5,10,11]: # rib4, rib8
            # if i in [4,5,6,7]: # rib4, rib6
            # if i in [10,11]: # rib6, rib8
            # if i != len(colors):
                plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)


        # plt.scatter(output_tsne[13*a:14*a, 0], output_tsne[13*a:14*a, 1], color=colors[2], alpha=0.5)
        plt.savefig(save_name+".png")
        plt.close() 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the properties of interpolation, extrapolation and triangulation.')
    parser.add_argument('--image_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    parser.add_argument('--pretrain_mode', type=str, choices=['LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X','lvmmed', 'ark'], default='EVA-X', help="Choose the pretraining mode")
    
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/sslgenesis/fromscratch_extrap_shuffle_compdecomp_consis/checkpoint.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/unique_multi_consis_compdecomp/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/ACE_v2/large_swinv2_fromIN_unique_multiscale_consis_compdecomp/checkpoint0020.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/CheSS/chess.pth.tar',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt',  help='The root dir of model.')
    
    parser.add_argument('--embd_dir', type=str, default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/Landmark_embd',  help='key image embeddings saving dictionary.')
    parser.add_argument('--test_list', type=str, default='./Landmark_Annotation', help='key image embeddings saving dictionary.')
    parser.add_argument('--save_file', type=str, default='./symmetry_similarity/ACEv2_swinv2_symmetry_Lclavicle_synvsgt.csv', help='the similarity save file')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    parser.add_argument('--backbone', type=str, default='swinv1', help='testing backbone')
    parser.add_argument('--img_size', type=int, default=224, help='image size, chess 512, eva-x 224')
    parser.add_argument('--shift', type=str, default=0,  help='compute embedding stride in 1024*1024 size image')
    parser.add_argument('--embd_dim', type=int, default=768,  help='lvmmed,chess:2048, swin:1024, vit:768')
    parser.add_argument('--add_syn', type=bool, default=False, help='image size')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps', 'ACE-v2', 'ark']:
        if args.backbone == 'swinv1':
            model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
        elif args.backbone == 'swinv2':
            model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                                        num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'CheSS':
        model = resnet50(num_classes=2)
    elif args.pretrain_mode == 'adamv2':
        model = convnext.__dict__['convnext_base']()
    elif args.pretrain_mode == 'RAD-DINO':
        model = AutoModel.from_pretrained('/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model
    elif args.pretrain_mode == 'EVA-X':
        model = eva_x_base_patch16(pretrained = args.model_path) # eva-x
    
    
    tsne(model=model, model_path=args.model_path, save_name="./images/EVA-X_symmetry_8points_changecolor", args=args)

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
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/sslgenesis/fromscratch_extrap_popar_compdecomp_consis/checkpoint0150.pth',save_name="./images/symmetry_4points3_fromscratch_extrap_popar_compdecomp_consis_150epc",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/patch_shuffling/pretrained_weight/patch_shuffle_student_teacher/checkpoint0100.pth',save_name="./images/tsne_plot_shuffle_localconsis",flip=0)
# tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/adam/Adam-v2_convnext_base.pth',save_name="./images/tsne_plot_adamv2",flip=0)
# tsne(model_path='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_mlp_overlapglobal/checkpoint0050.pth',save_name="./images/from_imagenet_matrixcompdecomp_mlp_overlapglobal_50epc_9cluster",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/sslgenesis_ablation/symmetry/pretrained_weight/symmetry_local/checkpoint0050.pth',save_name="./images/symmetry_local_rib8",flip=0)
# tsne(model_path='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',save_name="./images/ACEv2_swinv1_symmetry_8points_spinousprocess",flip=0)
