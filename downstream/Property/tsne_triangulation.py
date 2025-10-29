# visualize the t-sne of the ground-truth features and synthetic features using triangulation

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


# Directory with the text files
text_files_dir = './Landmark_Annotation'
# Directory with the png files
images_dir = '/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/'
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


def find_intersection(A, B, C, P): # compute A1
    import numpy as np
    
    # 解析直线方程的系数
    def line_coefficients(x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1
        return A, B, C

    # 计算AP的系数
    A1, B1, C1 = line_coefficients(A[0], A[1], P[0], P[1])
    # 计算BC的系数
    A2, B2, C2 = line_coefficients(B[0], B[1], C[0], C[1])

    # 构建系数矩阵和常数向量
    coefficients = np.array([[A1, B1], [A2, B2]])
    constants = np.array([C1, C2])

    # 使用numpy求解线性方程组
    intersection = np.linalg.solve(coefficients, constants)
    # ipdb.set_trace()
    return intersection

def get_embd(model, position, image, device, args=None):
    """
    get the embedding of one position
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    patch = crop_and_pad(image, position, (args.img_size, args.img_size))
    patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
    # patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_CUBIC)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = Image.fromarray(patch)

    patch = transform(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract features using the model
        # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
        _, features = model.forward_features(patch)
        # features = model(patch) # hugging face
        # features = features.last_hidden_state[:,1:] # hugging face
        # ipdb.set_trace()

        if args is None:
            f1 = features[:,90]
        else:
            if args.backbone == 'swinv1':
                f1 = features[:,90] # 90 for swinv1 backbone, 119 for swinv2 and 91 for vit backbone
            elif args.backbone == 'swinv2':
                f1 = features[:,119]
        
        # f1 = features # 90 for convnext

    return f1



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


def triangulation(A,B,C,P, image, model, device, args=None):
    # Generate a random triangle ABC
    # A, B, C = generate_random_triangle()
    # print(A,B,C)

    # # Generate a random point P inside the triangle ABC using barycentric coordinates
    # r1, r2 = np.random.rand(2)
    # if r1 + r2 > 1:
    #     r1, r2 = 1 - r1, 1 - r2
    # P = (1 - r1 - r2) * A + r1 * B + r2 * C
    # ipdb.set_trace()
    # Find intersection points
    A1 = find_intersection(A,B,C,P)
    B1 = find_intersection(B,A,C,P)
    C1 = find_intersection(C,A,B,P)
    # print(A1,B1,C1,P)

    # ipdb.set_trace()
    embd_P = get_embd(model, (round(P[0]), round(P[1])), image, device, args).cpu().numpy()
    embd_A = get_embd(model, A, image, device, args).cpu().numpy()
    embd_B = get_embd(model, B, image, device, args).cpu().numpy()
    embd_C = get_embd(model, C, image, device, args).cpu().numpy()
    # embd_A1 = get_embd(model, (round(A1[0]), round(A1[1])), image, device).cpu().numpy()
    # embd_B1 = get_embd(model, (round(B1[0]), round(B1[1])), image, device).cpu().numpy()
    # embd_C1 = get_embd(model, (round(C1[0]), round(C1[1])), image, device).cpu().numpy()

    if A1[0]>0 and A1[1]>0:
        t_a1 = (B[0]-A1[0])/(B[0]-C[0]) if (B[0]-C[0])!=0 else (B[1]-A1[1])/(B[1]-C[1])
        # print(t_a1)
        embd_A1 = t_a1*embd_C+(1-t_a1)*embd_B
        # using A1,A to predict P
        t = (A[0]-P[0])*1.0/(A[0]-A1[0])
        embd_P1_pred = t*embd_A1+(1-t)*embd_A
        similarity1 = cosine_similarity(embd_P.reshape(1, -1), embd_P1_pred.reshape(1, -1))
        return embd_P, embd_P1_pred, similarity1[0,0]

    if B1[0]>0 and B1[1]>0:
        t_b1 = (A[0]-B1[0])/(A[0]-C[0]) if (A[0]-C[0])!=0 else (A[1]-B1[1])/(A[1]-C[1])
        # print(t_b1)
        embd_B1 = t_b1*embd_C+(1-t_b1)*embd_A
        # using B1,B to predict P
        t = (B[0]-P[0])*1.0/(B[0]-B1[0])
        embd_P2_pred = t*embd_B1+(1-t)*embd_B
        similarity2 = cosine_similarity(embd_P.reshape(1, -1), embd_P2_pred.reshape(1, -1))
        return embd_P, embd_P2_pred, similarity2[0,0]

    if C1[0]>0 and C1[1]>0:
        t_c1 = (A[0]-C1[0])/(A[0]-B[0]) if (A[0]-B[0])!=0 else (A[1]-C1[1])/(A[1]-B[1])
        # print(t_c1)
        embd_C1 = t_c1*embd_B+(1-t_c1)*embd_A
        # using C1,C to predict P
        t = (C[0]-P[0])*1.0/(C[0]-C1[0])
        embd_P3_pred = t*embd_C1+(1-t)*embd_C
        similarity3 = cosine_similarity(embd_P.reshape(1, -1), embd_P3_pred.reshape(1, -1))
        return embd_P, embd_P3_pred, similarity3[0,0]

    







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
    output = []
    output1, output2, output3, output4, output5 = [], [], [], [], []
    similarities = []
    colors = plt.cm.get_cmap('tab20', 12)
    bgr_colors = [tuple(list(map(int, np.array(colors(i)[:3]) * 255))[::-1]) for i in range(12)]
    # flip = 0
    # positions = [2,10,18,34,42,50,21]# [2,10,18,34,42,50,21] [21,2,10,18] [21,34,42,50]
    # positions = [2,34,21,24,10,44,54,53,30]
    positions = [2,34,10,42,8,40,12,44,14,46,16,48,21]
    # positions = [2,34,10,42,16,48]
    # positions = [10,42,8,40,12,44,14,46,16,48]
    # positions = [10]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [position - 1 for position in positions]

    # write file
    file = open(args.save_file, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['image_name', 'similarity'])  # 写入表头
    file.flush()
    
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

                # selected_coord = coords[pos]
                selected_coord_A = coords[selected_positions[-1]] # spinous process
                selected_coord_B = coords[selected_positions[-3]] # left rib8
                selected_coord_C = coords[selected_positions[-2]] # right rib8
                selected_coord_P = coords[selected_positions[0]] # predict point: left clavicle

                # Read the image
                img = cv2.imread(os.path.join(images_dir, image_name))
                gt_embd_P, predict_embed_P, similarity = triangulation(selected_coord_A, selected_coord_B, selected_coord_C, selected_coord_P, img, model, device, args)
                similarities.append(similarity)
                writer.writerow([image_name, similarity])
                print(f"Processing {file_name}: similarity={similarity:.4f}")
                
                for selected_coord, idx in zip([selected_coord_A, selected_coord_B, selected_coord_C], [0,1,2]):
                    patch = crop_and_pad(img, selected_coord, (args.img_size, args.img_size))
                    patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)

                    patch = transform(patch).unsqueeze(0).to(device)
                    #print(patch.shape)
                    with torch.no_grad():
                        # Extract features using the model
                        # features = model.forward_features(patch) # timm
                        _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]

                        if args.backbone == 'swinv1':
                            f1 = features[:,90] # 90 for swinv1 backbone, 119 for swinv2 and 91 for vit backbone
                        elif args.backbone == 'swinv2':
                            f1 = features[:,119]

                        features = f1
                    if idx == 0:
                        output1.append(features.cpu().numpy())
                    elif idx == 1:
                        output2.append(features.cpu().numpy())  
                    elif idx == 2:
                        output3.append(features.cpu().numpy())

                output4.append(gt_embd_P)
                output5.append(predict_embed_P)
            
        # except:
        #     print(file_name)
        #     continue
    file.flush()
    # sys.exit(1)
    output = output1 + output2 + output3 + output4 + output5
    # a=len(output)//len(positions)
    a = len(output)//5
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
    if positions == [21,2,10,18]:
        colors = ['brown','blue', 'red', 'green']
    elif positions == [21,34,42,50]:
        colors = ['brown','yellow', 'orange', 'purple']
    elif positions == [2,34,21,24,10,44,54,53,30]:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    else:
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink','magenta','cyan','slategray','sienna','skyblue','turquoise']
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
    # colors = ['red']
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # Create a scatter plot

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
        # for i in range(len(positions)):
        #     if i in [0,1,4,5,6,7,10,11,12]: # 8 points+spinous process
        #     # if i in [0,1,4,5,6,7,10,11]: # 8 points
        #     # if i in [0,1,4,5,10,11]: # 6 points
        #     # if i in [0,1,4,5]: # clavicel, rib4
        #     # if i in [4,5,10,11]: # rib4, rib8
        #     # if i in [4,5,6,7]: # rib4, rib6
        #     # if i in [10,11]: # rib6, rib8
        #     # if i != len(colors):
        #         plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
        #         # plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], c=labels, alpha=0.5)
        #         # Use triangle marker for i = 7

        #     # else:
        #     #     plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5, marker='^')
        plt.scatter(output_tsne[0:a, 0], output_tsne[0:a, 1], color=colors[12], alpha=0.5) #  Spinous process
        plt.scatter(output_tsne[a:2*a, 0], output_tsne[a:2*a, 1], color=colors[10], alpha=0.5) #  left rib8
        plt.scatter(output_tsne[2*a:3*a, 0], output_tsne[2*a:3*a, 1], color=colors[11], alpha=0.5) #  right rib8
        plt.scatter(output_tsne[3*a:4*a, 0], output_tsne[3*a:4*a, 1], color=colors[0], alpha=0.5) # gt left clavicle, blue
        plt.scatter(output_tsne[4*a:5*a, 0], output_tsne[4*a:5*a, 1], color=colors[2], alpha=0.5) # predict left clavicle, green

        
        plt.savefig(save_name+".png")
        plt.close() 
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the properties of interpolation, extrapolation and triangulation.')
    parser.add_argument('--image_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/sslgenesis/fromscratch_extrap_shuffle_compdecomp_consis/checkpoint.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/unique_multi_consis_compdecomp/checkpoint0150.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/ACE_v2/large_swinv2_fromIN_unique_multiscale_consis_compdecomp/checkpoint0020.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/simmim/ckpt_epoch_100.pth',  help='The root dir of model.')
    parser.add_argument('--embd_dir', type=str, default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/Landmark_embd',  help='key image embeddings saving dictionary.')
    parser.add_argument('--test_list', type=str, default='./Landmark_Annotation', help='key image embeddings saving dictionary.')
    parser.add_argument('--ratio', type=float, default=0.75,  help='ration of OA/AB')
    parser.add_argument('--save_file', type=str, default='./interpolation/ACEv2_swinv2_leftclavicle_triangulation.csv', help='the similarity save file')
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
    
    tsne(device, model=model, model_path=args.model_path, save_name="./images/ACEv2_swinv2_triangulation_left_clavicle",flip=0, args=args)

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
