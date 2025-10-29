import numpy as np
import torch
import torchvision
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
            patch = crop_and_pad(image, center, (96,96), 0) # crop size of 96*96 image around the center

            patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = Image.fromarray(patch)

            transforms = transform()

            patch = transforms(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract features using the model
                
                if args.pretrain_mode in ['CheSS']: # chess
                    patch = patch[:,0].unsqueeze(1)
                # ipdb.set_trace()
                if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS','lvmmed']:
                    _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
                # ipdb.set_trace()
                elif args.pretrain_mode in ['RAD-DINO']:
                    features = model(patch) # hugging face
                    features = features.last_hidden_state[:,1:] # hugging face
                elif args.pretrain_mode in ['adamv2']:
                    features = model.extract_features(patch) # convnext
                # print(features.shape)
                # f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                f1 = torch.mean(features, dim=1)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--image_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    parser.add_argument('--pretrain_mode', type=str, choices=['LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X','lvmmed'], default='adamv2', help="Choose the pretraining mode")
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/extrapolation/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/hierar_compdecomp/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/extrapolation_MIM/checkpoint0100_MIM.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/large_fromscratch_extrap_shuffle_compdecomp_consis/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/fromscratch_extrap_popar_compdecomp_consis/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/CheSS/chess.pth.tar',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/LeADER/LeADER_swin_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/large_fromMIM_extrap_shuffle_consis_compdecomp/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/lvmmed/lvmmed_resnet.torch',  help='The root dir of model.')
    
    parser.add_argument('--query_list', type=str, default='./correspondence/query_key_list/landmark_query_list.txt',  help='query list')
    parser.add_argument('--key_list', type=str, default='./correspondence/query_key_list/landmark_key_list.txt',  help='query image.')
    parser.add_argument('--text_files_dir', type=str, default='./Landmark_Annotation',  help='test annotation dictionary.')
    parser.add_argument('--embd_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_testadamv2_100imgs_crop96global',  help='key image embeddings saving dictionary.')
    parser.add_argument('--error_file', type=str, default='./correspondence/error_testadamv2_100imgs_crop96global.csv',  help='save the errors of each predicted landmark.')
    parser.add_argument('--stride', type=str, default=16,  help='compute embedding stride in 1024*1024 size image')
    parser.add_argument('--embd_dim', type=int, default=1024,  help='lvmmed,chess:2048, swin:1024, vit:768')
    parser.add_argument('--img_size', type=int, default=224,  help='resize resolution')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    args = parser.parse_args()
    

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps']:
        model = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'CheSS':
        model = resnet50(num_classes=2)
    elif args.pretrain_mode == 'adamv2':
        model = convnext.__dict__['convnext_base']()
    elif args.pretrain_mode == 'RAD-DINO':
        model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model
    elif args.pretrain_mode == 'EVA-X':
        model = eva_x_base_patch16(pretrained = args.model_path) # eva-x
    elif args.pretrain_mode == 'lvmmed':
        model = torchvision.models.resnet50()
        # model = resnet50(num_classes=2)


    checkpoint = torch.load(args.model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        if args.pretrain_mode == 'adamv2':
            checkpoint = checkpoint['teacher']
        else:
            checkpoint = checkpoint['student']
        
    except:
        checkpoint = checkpoint
        # checkpoint = checkpoint['model']
        # checkpoint = checkpoint['state_dict']
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
    positions = [2,34,29,24,40,10,30,25,46,16,32,54,53] # query positions
    selected_positions = [position - 1 for position in positions]

    
    # save the error of each predicted position
    file =  open(args.error_file, mode='w', newline='', encoding='utf-8')
    # landmark_name = ['ImgName', 'Right clavicle', 'Left clavicle','Spinous process','Azygos arch','Right rib5','Left rib6',\
                    # 'Right hemidiaphragm','Left hemidiaphragm','Pulmonary artery','Right rib7','Aortic arch','Left ventricle boarder','Left rib3']
    landmark_name = ['QueryImage', 'Right clavicle', 'Left clavicle','Aortic Arch','Azygos Arch','Rib4','Rib5',\
                    'Pulmonary Artery','Right Hilum','Rib7','Rib8','Left Ventricle Border','Right Hemidiaphragm','Left Hemidiaphragm']
    writer = csv.writer(file)
    writer.writerow(landmark_name)  # 写入表头
    file.flush()
    # for row in data_dicts:
    #     writer.writerow(row)


    

    # generate the embedding books for key images
    if not os.path.exists(args.embd_dir):
        os.makedirs(args.embd_dir)  

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
        coords_query = {int(coord.split(',')[2]): (int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_query if coord != ''}
        
        error_row = []
        pos_num = 0
        for pos in selected_positions: # 13 landmarks
            pos_num += 1
            if pos in coords_query:
                selected_coord = coords_query[pos]
            else:
                continue
            
            patch = crop_and_pad(query_image, selected_coord, (96,96), 0)
            patch = cv2.resize(patch, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            patch = Image.fromarray(patch)

            patch = transf(patch).unsqueeze(0).to(device)
            # ipdb.set_trace()
            with torch.no_grad():
                # Extract features using the model
                if args.pretrain_mode in ['CheSS']: # chess
                    patch = patch[:,0].unsqueeze(1)
                if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS','lvmmed']:
                    _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
                # ipdb.set_trace()
                elif args.pretrain_mode in ['RAD-DINO']:
                    features = model(patch) # hugging face
                    features = features.last_hidden_state[:,1:] # hugging face
                elif args.pretrain_mode in ['adamv2']:
                    features = model.extract_features(patch) # convnext
                # print(features.shape)
                # f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
                f1 = torch.mean(features, dim=1)

            f1 = f1.cpu().numpy()

            key_pred_position = [] # save the most similar embedding's positions
            key_gt_position = [] # len=100
            key_num = 0
            for key in keydict.keys(): # 100 key images
                key_num += 1
                print(f'computing query {query_num}, position {pos_num}, key {key_num}...')
                # key_features = torch.load(os.path.join(args.embd_dir, key+'.npy'))
                key_features = np.load(os.path.join(args.embd_dir, key+'.npy'))

                key_label_txt = open(os.path.join(args.text_files_dir, keydict[key]), 'r').read().strip()
                key_image_name, *coords_key = key_label_txt.split('#')
                coords_key = {int(coord.split(',')[2]): (int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_key if coord != ''}
                
                if pos in coords_key:
                    key_gt_position.append(coords_key[pos])
                else:
                    continue

                similarities = np.zeros((w_embd, h_embd))
                for i in range(w_embd):
                    for j in range(h_embd):
                        similarities[i, j] = cosine_similarity(f1, key_features[i, j].reshape(1, -1)).item()

                max_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
                x_pred, y_pred = int(max_index[0]*args.stride+args.stride//2), int(max_index[1]*args.stride+args.stride//2)
                print((x_pred, y_pred), coords_key[pos])
                key_pred_position.append((x_pred, y_pred))

            error_100key = 0
            for i in range(len(key_pred_position)):
                error = math.sqrt((key_pred_position[i][0]-key_gt_position[i][0])**2+(key_pred_position[i][1]-key_gt_position[i][1])**2)
                error_100key += error
            error_100key = error_100key/len(key_pred_position)
            error_row.append(error_100key)
        
        error_row.insert(0, query_img)
        writer.writerow(error_row)
        print(error_row)
        file.flush()








