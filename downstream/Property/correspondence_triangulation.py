# Using the synthetic embedding to retrieve the correspondent landmarks. If the synthetic embedding can successfully
# retrieve the landmarks, then the synthetic embedding is considered to be a similar representation of the original embedding.


import numpy as np
import torch
import ipdb
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
from models.resnet import resnet50
from models.eva_x import eva_x_base_patch16
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image, ImageOps
import torch.nn as nn
from sklearn.cluster import KMeans
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import math
import csv
import models.convnext as convnext
from transformers import AutoModel
import time



def crop_and_pad(image, center, size=(96, 96), stride=6.86):
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


def transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            ])
    return transform


def save_embeddings(args, model, image, device, stride, embed_dim=1024):
    """
    input the image with size 1024*1024,extract embeddings for each 32*32 patch
    return patch embedding of the whole image, the number is 128*128=16384
    """
    
    w0,h0 = image.shape[:2]
    w_embd, h_embd = w0//stride, h0//stride
    embeddings = torch.zeros((w_embd,h_embd,embed_dim))

    for i in range(w_embd):
        for j in range(h_embd):
            center = (stride/2+stride*i, stride/2+stride*j) # center coordinate for each patch
            # patch = crop_and_pad(image, center, (stride*14,stride*14), stride) # crop size of 448*448 image around the center
            if args.pretrain_mode in ['PEAC','ACE','Lamps', 'ACE-v2']: # imgsize 448
                shift = 32
            elif args.pretrain_mode in ['LeADER', 'adamv2','EVA-X','RAD-DINO']: # imgsize 224
                shift = 0
            elif args.pretrain_mode in ['CheSS']:
                shift = 28
            patch = crop_and_pad(image, center, (args.crop_size,args.crop_size), shift) # crop size of 96*96 image around the center

            patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = Image.fromarray(patch)

            transforms = transform()

            patch = transforms(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract features using the model
                
                if embed_dim == 2048: # chess
                    patch = patch[:,0].unsqueeze(1)
                # ipdb.set_trace()
                if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS', 'ACE-v2']:
                    _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
                # ipdb.set_trace()
                elif args.pretrain_mode in ['RAD-DINO']:
                    features = model(patch) # hugging face
                    features = features.last_hidden_state[:,1:] # hugging face
                elif args.pretrain_mode in ['adamv2']:
                    features = model.extract_features(patch) # convnext

                # ipdb.set_trace()
                if args.backbone == 'swinv2':
                    f1 = features[:,119]
                else:
                    if args.pretrain_mode in ['PEAC','ACE','Lamps','EVA-X', 'ACE-v2']:
                        f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                    elif args.pretrain_mode in ['LeADER', 'adamv2']:
                        f1 = features[:,24]
                    elif args.pretrain_mode in ['RAD-DINO']:
                        f1 = features[:,684] # rad-dino has 1369(37*37) features
                    elif args.pretrain_mode in ['CheSS']:
                        f1 = features[:,119] # CheSS has 256(16*16) features
                    

            embeddings[i,j] = f1
            print(w_embd*i+j)

    embeddings = embeddings.cpu().numpy()

    return embeddings


def draw_circle(image, points, gt_points, colors, stride):
    color_dict = {
    'blue': (255, 0, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'magenta': (255, 0, 255),
    'pink': (203, 192, 255),
    'cyan': (255, 255, 0),
    'tan': (180, 229, 255),
    'thistle': (216, 191, 216),
    'honeydew': (240, 255, 240),
    'orchild': (214, 112, 218)
    }

    size = 22 # 半径
    # 叉的大小
    cross_size = 20
    for point, gt_point, color_name in zip(points, gt_points, colors):
        # x = point[0]*stride+stride//2
        # y = point[1]*stride+stride//2
        x = point[0]
        y = point[1]
        x_gt, y_gt = gt_point
        color = color_dict[color_name]
        
        cv2.circle(image, (x_gt,y_gt), size, color, -1)  # -1 表示填充圆 gt
        cv2.line(image, (x - cross_size, y - cross_size), (x + cross_size, y + cross_size), color_dict['red'], 8) # prediction
        cv2.line(image, (x - cross_size, y + cross_size), (x + cross_size, y - cross_size), color_dict['red'], 8)
    return image


def get_embd(model, position, image, device, args=None):
    """
    get the embedding of one position
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    patch = crop_and_pad(image, position, (args.img_size, args.img_size), stride=32)
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



def calculate_point_B(A, P, ratio):
    """
    根据A点、P点坐标和比例ratio计算B点坐标
    
    参数：
    A -- 起点A的坐标，格式为(x, y)
    P -- 中间点P的坐标，格式为(x, y)
    ratio -- AP与AB的比例，0 < ratio < 1
    
    返回：
    B点坐标 (x, y)
    """
    # 如果比例为0或1的特殊情况处理
    if ratio == 0:
        return A  # 如果比例是0，则B点就是A点
    elif ratio == 1:
        return P  # 如果比例是1，则B点就是P点
    
    # 解向量方程：
    # 设A(xa, ya), B(xb, yb), P(xp, yp)
    # 比例关系：|AP| / |AB| = ratio
    # 向量关系：P = A + ratio * (B - A)
    
    # 从上述等式推导：
    # P = A + ratio * (B - A)
    # P = (1 - ratio) * A + ratio * B
    # ratio * B = P - (1 - ratio) * A
    # B = [P - (1 - ratio) * A] / ratio
    
    Ax, Ay = A
    Px, Py = P
    
    # 计算B点坐标
    Bx = int((Px - (1 - ratio) * Ax) / ratio)
    By = int((Py - (1 - ratio) * Ay) / ratio)
    
    return (Bx, By)


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--image_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    parser.add_argument('--pretrain_mode', type=str, choices=['LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X', 'ACE-v2'], default='ACE-v2', help="Choose the pretraining mode")
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/extrapolation/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/hierar_compdecomp/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/extrapolation_MIM/checkpoint0100_MIM.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/large_fromscratch_extrap_shuffle_compdecomp_consis/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/fromscratch_extrap_popar_compdecomp_consis/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/CheSS/chess.pth.tar',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/LeADER/LeADER_swin_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/large_fromMIM_extrap_shuffle_consis_compdecomp/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    
    parser.add_argument('--ratio', type=float, default=0.25,  help='ration of OA/AB')
    parser.add_argument('--query_list', type=str, default='./correspondence/query_key_list/landmark_query_list.txt',  help='query list')
    parser.add_argument('--key_list', type=str, default='./correspondence/query_key_list/landmark_key_list.txt',  help='query image.')
    parser.add_argument('--text_files_dir', type=str, default='./Landmark_Annotation',  help='test annotation dictionary.')
    parser.add_argument('--embd_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_fromIN_ACEv2_swinv2_100imgs_crop448local',  help='key image embeddings saving dictionary.')
    # parser.add_argument('--embd_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_fromIN_unique_multiscale_consis_compdecomp_100imgs_crop448local',  help='key image embeddings saving dictionary.')
    parser.add_argument('--error_file', type=str, default='./correspondence/error_ACEv2_swinv2_triangulation_pulmonary_artery.csv',  help='save the errors of each predicted landmark.')
    parser.add_argument('--stride', type=str, default=16,  help='compute embedding stride in 1024*1024 size image')
    parser.add_argument('--embd_dim', type=int, default=1024,  help='save the key embeddings of the whole image,768 for eva-x')
    parser.add_argument('--backbone', type=str, default='swinv2', help='testing backbone')
    parser.add_argument('--img_size', type=int, default=512,  help="the model's pretrain image size")
    parser.add_argument('--crop_size', type=int, default=512,  help="crop size of each landmark")
    parser.add_argument('--device', type=str, default='2',  help='device number')
    args = parser.parse_args()
    

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.backbone == 'swinv2':
        model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                                    num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    else:
        if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps', 'ACE-v2']:
            model = SwinTransformer(img_size=args.img_size,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                                    num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
        elif args.pretrain_mode == 'CheSS':
            model = resnet50(num_classes=2)
        elif args.pretrain_mode == 'adamv2':
            model = convnext.__dict__['convnext_base']()
        elif args.pretrain_mode == 'RAD-DINO':
            model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model
        elif args.pretrain_mode == 'EVA-X':
            model = eva_x_base_patch16(pretrained = args.model_path) # eva-x
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
        # checkpoint = checkpoint['teacher']
    except:
        checkpoint = checkpoint
        if args.pretrain_mode in ['CheSS']:
            
        # checkpoint = checkpoint['model']
            checkpoint = checkpoint['state_dict']
    #checkpoint = checkpoint['student']
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

    model.to(device)

    

    transf = transform()
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    # positions = [2,34,21,24,10,44,54,53,30] # query positions
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    # positions = [2,10,18,34,42,50,21]
    colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan', 'tan', 'thistle','honeydew','orchild']
    # positions = [2,34,21,24,10,44,54,53,30,14,29,31,38] # query positions
    # positions = [2,34,29,24,40,10,30,25,46,16,32,54,53] # query positions
    # positions = [2,34] # query positions| left and right clavicle
    # positions = [8,12] # query positions| left rib4 and left rib6
    positions = [2,34,10,42,8,40,12,44,14,46,30,16,48,21]
    selected_positions = [position - 1 for position in positions]

    
    # save the error of each predicted position
    file =  open(args.error_file, mode='w', newline='', encoding='utf-8')
    # landmark_name = ['ImgName', 'Right clavicle', 'Left clavicle','Spinous process','Azygos arch','Right rib5','Left rib6',\
                    # 'Right hemidiaphragm','Left hemidiaphragm','Pulmonary artery','Right rib7','Aortic arch','Left ventricle boarder','Left rib3']
    # landmark_name = ['QueryImage', 'Right clavicle', 'Left clavicle','Aortic Arch','Azygos Arch','Rib4','Rib5',\
    #                 'Pulmonary Artery','Right Hilum','Rib7','Rib8','Left Ventricle Border','Right Hemidiaphragm','Left Hemidiaphragm']
    # landmark_name = ['QueryImage', 'Right clavicle', 'Left clavicle']
    landmark_name = ['QueryImage', 'Error predict embd', 'Error gt embd', 'Similarity']
    writer = csv.writer(file)
    writer.writerow(landmark_name)  # 写入表头
    file.flush()
    # for row in data_dicts:
    #     writer.writerow(row)


    

    # generate the embedding books for key images
    if not os.path.exists(args.embd_dir):
        os.makedirs(args.embd_dir)  # 递归创建目录

    keydict = {} # image name: landmarks label name
    with open(args.key_list, 'r') as f:
        for line in f:
            keydict[line.split('-')[0]+'.png'] = line.split('\n')[0]
    print(keydict)

    for key in keydict.keys(): # traversal of the 100 key images
        key_image = cv2.imread(os.path.join(args.image_dir, key))
        if not os.path.exists(os.path.join(args.embd_dir, key+'.npy')):# save key image features
            key_features = save_embeddings(args, model, key_image, device, args.stride, args.embd_dim)
            # torch.save(key_features, os.path.join(args.embd_dir, key)+'.npy')
            np.save(os.path.join(args.embd_dir, key)+'.npy', key_features)
        

    # traversal of the 100 query images
    querydict = {} # image name: landmarks label name
    with open(args.query_list, 'r') as f:
        for line in f:
            querydict[line.split('-')[0]+'.png'] = line.split('\n')[0]
    print(querydict)

    query_num = 0
    for query_img in querydict.keys(): # 100 query images
        
        query_num += 1
        query_image = cv2.imread(os.path.join(args.image_dir, query_img))
        
        w0,h0,_ = query_image.shape
        w_embd, h_embd = w0//args.stride, h0//args.stride # saving embd number: w_embd*h_embd

        query_label_txt = open(os.path.join(args.text_files_dir, querydict[query_img]), 'r').read().strip()

        query_image_name, *coords_query = query_label_txt.split('#')
        coords_query = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_query if coord != '']
        
        error_row = []
        pos_num = 0
        # for pos in selected_positions: # 13 landmarks
        
        time_start = time.time()
        pos_num += 1
        # if pos in coords_query:
        #     selected_coord = coords_query[pos]
        # else:
        #     continue
        # selected_coord_A = coords_query[selected_positions[0]] # left rib4
        # selected_coord_P = coords_query[selected_positions[1]] # left rib6
        
        try:
            selected_coord_A = coords_query[selected_positions[-1]] # spinous process
            selected_coord_B = coords_query[selected_positions[-3]] # left rib8
            selected_coord_C = coords_query[selected_positions[-2]] # right rib8
            # selected_coord_P = coords_query[selected_positions[0]] # predict point: left clavicle
            selected_coord_P = coords_query[selected_positions[-4]] # predict point: pulmonary artery
            
            gt_embd_P, predict_embed_P, similarity = triangulation(selected_coord_A, selected_coord_B, selected_coord_C, selected_coord_P, query_image, model, device, args)
            print(similarity)
            # ipdb.set_trace()
        except:
            continue
        

        key_pred_position = [] # save the most similar embedding's positions | predict embeding P
        key_pred_position2 = [] # save the most similar embedding's positions | gt embeding P
        key_gt_position = [] # len=100
        key_num = 0
        for key in keydict.keys(): # 100 key images
            key_num += 1
            print(f'computing query {query_num}, position {pos_num}, key {key_num}...')
            # key_features = torch.load(os.path.join(args.embd_dir, key+'.npy'))
            key_features = np.load(os.path.join(args.embd_dir, key+'.npy'))

            key_label_txt = open(os.path.join(args.text_files_dir, keydict[key]), 'r').read().strip()
            key_image_name, *coords_key = key_label_txt.split('#')
            # coords_key = {int(coord.split(',')[2]): (int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_key if coord != ''}
            coords_key = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_key if coord != '']
            # if pos in coords_key:
            #     key_gt_position.append(coords_key[pos])
            # else:
            #     continue
            
            gt_position = coords_key[selected_positions[-4]]
            key_gt_position.append(gt_position) # gt position

            # using the predict embedding P to find the most similar position in key image
            similarities = np.zeros((w_embd, h_embd))
            for i in range(w_embd):
                for j in range(h_embd):
                    similarities[i, j] = cosine_similarity(predict_embed_P, key_features[i, j].reshape(1, -1)).item()

            max_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
            x_pred, y_pred = int(max_index[0]*args.stride+args.stride//2), int(max_index[1]*args.stride+args.stride//2)
            print((x_pred, y_pred), gt_position)
            key_pred_position.append((x_pred, y_pred))
            
            # using the gt embedding P to find the most similar position in key image
            similarities = np.zeros((w_embd, h_embd))
            for i in range(w_embd):
                for j in range(h_embd):
                    similarities[i, j] = cosine_similarity(gt_embd_P, key_features[i, j].reshape(1, -1)).item()

            max_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
            x_pred, y_pred = int(max_index[0]*args.stride+args.stride//2), int(max_index[1]*args.stride+args.stride//2)
            print((x_pred, y_pred), gt_position)
            key_pred_position2.append((x_pred, y_pred))


        error_100key = 0
        for i in range(len(key_pred_position)):
            error = math.sqrt((key_pred_position[i][0]-key_gt_position[i][0])**2+(key_pred_position[i][1]-key_gt_position[i][1])**2)
            error_100key += error
        error_100key = error_100key/100
        error_row.append(error_100key)
        
        
        error_100key = 0
        for i in range(len(key_pred_position2)):
            error = math.sqrt((key_pred_position2[i][0]-key_gt_position[i][0])**2+(key_pred_position2[i][1]-key_gt_position[i][1])**2)
            error_100key += error
        error_100key = error_100key/100
        error_row.append(error_100key)
        
        error_row.append(similarity)

        time_end = time.time()
        print('time cost ', time_end-time_start, 's')
        
        error_row.insert(0, query_img)
        writer.writerow(error_row)
        print(error_row)
        file.flush()






    # for t in range(5):


    #     query_image = cv2.imread(os.path.join(args.query_image, imglist[t]))
    #     w0,h0,_ = query_image.shape
    #     w_embd, h_embd = w0//args.stride, h0//args.stride # saving embd number: w_embd*h_embd

    #     key_imglist = [x for x in imglist if x!=imglist[t]]

    #     for j in key_imglist:
    #         key_pred_position = [] # save the most similar embedding's positions
    #         key_gt_position = []
        
    #         with open(os.path.join(args.text_files_dir, labeldir[j]), 'r') as g:
    #             content = g.read().strip()
    #             key_image_name, *coords_key = content.split('#')
    #             key_image_name = key_image_name.split('-')[0] + '.png'
    #             coords_key = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_key if coord != '']
                

    #             key_image = cv2.imread(os.path.join(args.query_image, j))
    #             if not os.path.exists(os.path.join(args.embd_dir, key_image_name+'.npy')):# save key image features                   
    #                 key_features = save_embeddings(model, key_image, device, args.stride, args.embd_dim)
    #                 torch.save(key_features, os.path.join(args.embd_dir, key_image_name)+'.npy')
    #             else: # load key image features
    #                 key_features = torch.load(os.path.join(args.embd_dir, key_image_name+'.npy'))

    #         print(t)
    #         with open(os.path.join(args.text_files_dir, labeldir[imglist[t]]), 'r') as f: # query image
    #             content = f.read().strip()
    #             # Split the content to get image name and coordinates
    #             image_name, *coords = content.split('#')
    #             # Parse the coordinates
    #             coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']

    #             for pos in selected_positions:
                    
    #                 key_gt_position.append(coords_key[pos])
    #                 selected_coord = coords[pos]

    #                 # patch = crop_and_pad(query_image, selected_coord, (args.stride*14,args.stride*14), stride=args.stride)
    #                 patch = crop_and_pad(query_image, selected_coord, (448,448), 32)
    #                 patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
    #                 patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    #                 patch = Image.fromarray(patch)

    #                 patch = transf(patch).unsqueeze(0).to(device)

    #                 with torch.no_grad():
    #                     # Extract features using the model
    #                     if args.embd_dim == 2048: # chess
    #                         patch = patch[:,0].unsqueeze(1)
    #                     # _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
    #                     features = model(patch) # hugging face
    #                     features = features.last_hidden_state[:,1:] # hugging face
    #                     # features = model.extract_features(patch) # convnext
    #                     # print(features.shape)
    #                     f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

    #                 f1 = f1.cpu().numpy()

    #                 similarities = np.zeros((w_embd, h_embd))
    #                 for i in range(w_embd):
    #                     for j in range(h_embd):
    #                         similarities[i, j] = cosine_similarity(f1, key_features[i, j].reshape(1, -1))

    #                 max_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
    #                 print(max_index)
    #                 x_pred, y_pred = int(max_index[0]*args.stride+args.stride//2), int(max_index[1]*args.stride+args.stride//2)
    #                 key_pred_position.append((x_pred, y_pred))


    #         key_image = draw_circle(key_image, key_pred_position, key_gt_position, colors, args.stride) # prediction
    #         cv2.imwrite(os.path.join(args.corres_target, key_image_name), key_image)

    #         # save the prediction position errors
    #         error_list = []
    #         for i in range(len(key_pred_position)):
    #             error = math.sqrt((key_pred_position[i][0]-key_gt_position[i][0])**2+(key_pred_position[i][1]-key_gt_position[i][1])**2)
    #             error_list.append(error)
    #         error_list.insert(0, key_image_name)

    #         writer.writerow(error_list)
    #         print(error_list)
    #         file.flush()







