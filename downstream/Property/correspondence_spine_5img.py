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
from models.resnet import resnet50
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
import matplotlib.pyplot as plt
import seaborn as sns



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


def transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            ])
    return transform


def save_embeddings(model, image, device, stride, embed_dim=1024):
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
            patch = crop_and_pad(image, center, (224,224), 16) # crop size of 448*448 image around the center

            patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = Image.fromarray(patch)

            transforms = transform()

            patch = transforms(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract features using the model
                
                if embed_dim == 2048: # chess
                    patch = patch[:,0].unsqueeze(1)
                # ipdb.set_trace()
                _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
                # features = model(patch) # hugging face
                # features = features.last_hidden_state[:,1:] # hugging face
                
                # features = model.extract_features(patch) # convnext
                # print(features.shape)
                f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

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



def draw_and_save(query_image, key_image,
                          query_gt_position, key_gt_box, key_pred_position,
                          save_path="result.jpg"):
    query_img_vis = query_image.copy()
    key_img_vis = key_image.copy()

    # 12种高对比颜色 (BGR 格式)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (0, 128, 255),    # Orange
        (203, 192, 255),  # Pink
        (0, 128, 0),      # Dark Green
        (128, 0, 0),      # Maroon
        (0, 0, 128),      # Navy
    ]

    n = len(query_gt_position)
    assert len(key_gt_box) == n and len(key_pred_position) == n, "All lists must have the same length."
    assert n <= 12, "This version supports up to 12 points with unique colors."

    for i in range(n):
        color = colors[i]

        # draw query point
        qx, qy = query_gt_position[i]
        cv2.circle(query_img_vis, (int(qx), int(qy)), radius=6, color=color, thickness=-1)

        # draw key box
        x, y, w, h = key_gt_box[i]
        cv2.rectangle(key_img_vis, (int(x), int(y)), (int(x + w), int(y + h)), color=color, thickness=2)

        # draw key predicted point
        px, py = key_pred_position[i]
        cv2.circle(key_img_vis, (int(px), int(py)), radius=6, color=color, thickness=-1)

    # 保证尺寸一致
    if query_img_vis.shape != key_img_vis.shape:
        h = min(query_img_vis.shape[0], key_img_vis.shape[0])
        w = min(query_img_vis.shape[1], key_img_vis.shape[1])
        query_img_vis = cv2.resize(query_img_vis, (w, h))
        key_img_vis = cv2.resize(key_img_vis, (w, h))

    # 拼接并保存
    combined = np.concatenate((query_img_vis, key_img_vis), axis=1)
    cv2.imwrite(save_path, combined)
    print(f"Saved result to {save_path}")


def point_to_bbox_edge_distance_outside(point, bbox):
    x, y, w, h = bbox
    px, py = point
    left = x
    right = x + w
    top = y
    bottom = y + h

    if left <= px <= right:
        # 垂直方向最近边（点在水平投影内）
        return min(abs(py - top), abs(py - bottom))
    elif top <= py <= bottom:
        # 水平方向最近边（点在垂直投影内）
        return min(abs(px - left), abs(px - right))
    else:
        # 在四角区域，距离为最近边界的水平或垂直距离
        d_top = abs(py - top)
        d_bottom = abs(py - bottom)
        d_left = abs(px - left)
        d_right = abs(px - right)
        return min(d_top, d_bottom, d_left, d_right)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--image_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/PAX_RAY/images_patlas/images_patlas/',  help='Dictionary of the image file.')
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
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/POPAR/POPAR_Swin_448.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/uniqueness/pretrained_weight/fromIN_uniqueness_multigranu_big_local_crop/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/uniqueness/pretrained_weight/uniqueness_multigranu_big_local_crop/checkpoint0150.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/consistency/pretrained_weight/fromIN_consis/checkpoint0150.pth',  help='The root dir of model.')
    
    parser.add_argument('--query_text_file', type=str, default='./Landmark_Annotation/00000020_001-gt-12_4-pa.txt',  help='query text file containing query points position.')
    parser.add_argument('--text_files_dir', type=str, default='./correspondence/PAX_RAY_spine_query_key/query',  help='test annotation dictionary.')
    parser.add_argument('--embd_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/PAX_RAY/correspond_embd/consistency_fromIN_150epc_5imgs',  help='key image embeddings saving dictionary.')
    parser.add_argument('--error_file', type=str, default='./correspondence/spinr_error_consistency_fromIN_150epc.csv',  help='save the errors of each predicted landmark.')
    parser.add_argument('--stride', type=str, default=4,  help='compute embedding stride in 512*512 size image')
    parser.add_argument('--save_embd', type=bool, default=False,  help='save the key embeddings of the whole image')
    parser.add_argument('--embd_dim', type=int, default=1024,  help='save the key embeddings of the whole image')
    parser.add_argument('--img_size', type=int, default=448,  help='save the key embeddings of the whole image')
    parser.add_argument('--device', type=str, default='1',  help='device number')
    args = parser.parse_args()
    

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer(img_size=args.img_size,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    # model = resnet50(num_classes=2)
    # model = convnext.__dict__['convnext_base']()
    # model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model

    checkpoint = torch.load(args.model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
        # checkpoint = checkpoint['teacher']
    except:
        # checkpoint = checkpoint
        checkpoint = checkpoint['model']
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
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan', 'tan', 'thistle','honeydew','orchild']
    # positions = [2,34,21,24,10,44,54,53,30,14,29,31,38] # query positions
    # selected_positions = [position - 1 for position in positions]

    
    # save the error of each predicted position
    file =  open(args.error_file, mode='w', newline='', encoding='utf-8')
    landmark_name = ['QueryImg', 'KeyImg', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    writer = csv.writer(file)
    writer.writerow(landmark_name)  # 写入表头
    file.flush()
    # for row in data_dicts:
    #     writer.writerow(row)

    imglist = ['0018a_frontal.png', '0060a_frontal.png', '0097a_frontal.png', '0171a_frontal.png', '0173a_frontal.png']
    labeldir = {'0018a_frontal.png':'0018a_frontal.png.txt', 
                '0060a_frontal.png':'0060a_frontal.png.txt', 
                '0097a_frontal.png':'0097a_frontal.png.txt', 
                '0171a_frontal.png':'0171a_frontal.png.txt', 
                '0173a_frontal.png':'0173a_frontal.png.txt'}


    if not os.path.exists(args.embd_dir):
        os.makedirs(args.embd_dir)  # 递归创建目录
        
    total_points = 0
    correct_match = 0
    non_correct_error = []


    for t in range(5):

        query_image_name = imglist[t]
        query_image = cv2.imread(os.path.join(args.image_dir, query_image_name))
        w0,h0,_ = query_image.shape
        w_embd, h_embd = w0//args.stride, h0//args.stride # saving embd number: w_embd*h_embd

        key_imglist = [x for x in imglist if x!=imglist[t]]

        for j in key_imglist:
            key_pred_position = [] # save the most similar embedding's positions
            key_gt_position = []
            key_gt_box = []
            query_gt_position = []
        
            with open(os.path.join(args.text_files_dir, labeldir[j]), 'r') as g:
                # content = g.read().strip()
                # key_image_name, *coords_key = content.split('#')
                # key_image_name = key_image_name.split('-')[0] + '.png'
                # coords_key = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords_key if coord != '']
                key_image_name = j
                for line_num, line in enumerate(g):
                    line = line.strip().split('[', 1)[1].split(']')[0]
                    line = line.split(',')
                    line = [float(x) for x in line]
                    x, y, width, height = line
                    selected_coord = (x + width / 2, y + height / 2)
                    key_gt_position.append(selected_coord)
                    key_gt_box.append((x, y, width, height))
                

                key_image = cv2.imread(os.path.join(args.image_dir, key_image_name))
                if not os.path.exists(os.path.join(args.embd_dir, key_image_name+'.npy')):# save key image features                   
                    key_features = save_embeddings(model, key_image, device, args.stride, args.embd_dim)
                    torch.save(key_features, os.path.join(args.embd_dir, key_image_name)+'.npy')
                else: # load key image features
                    key_features = torch.load(os.path.join(args.embd_dir, key_image_name+'.npy'))

            print(t)
            with open(os.path.join(args.text_files_dir, labeldir[query_image_name]), 'r') as f: # query image
                for line_num, line in enumerate(f):
                    line = line.strip().split('[', 1)[1].split(']')[0]
                    line = line.split(',')
                    line = [float(x) for x in line]
                    x, y, width, height = line
                    selected_coord = (x + width / 2, y + height / 2)
                    query_gt_position.append(selected_coord)
                # content = f.read().strip()
                # # Split the content to get image name and coordinates
                # image_name, *coords = content.split('#')
                # # Parse the coordinates
                # coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']

                # for pos in selected_positions:
                    
                #     key_gt_position.append(coords_key[pos])
                #     selected_coord = coords[pos]

                    # patch = crop_and_pad(query_image, selected_coord, (args.stride*14,args.stride*14), stride=args.stride)
                    patch = crop_and_pad(query_image, selected_coord, (224,224), 16)
                    patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)

                    patch = transf(patch).unsqueeze(0).to(device)

                    with torch.no_grad():
                        # Extract features using the model
                        if args.embd_dim == 2048: # chess
                            patch = patch[:,0].unsqueeze(1)
                        _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                        # features = model(patch) # hugging face
                        # features = features.last_hidden_state[:,1:] # hugging face
                        # features = model.extract_features(patch) # convnext
                        # print(features.shape)
                        f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

                    f1 = f1.cpu().numpy()

                    similarities = np.zeros((w_embd, h_embd))
                    for i in range(w_embd):
                        for j in range(h_embd):
                            similarities[i, j] = cosine_similarity(f1, key_features[i, j].reshape(1, -1))

                    max_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
                    print(max_index)
                    x_pred, y_pred = int(max_index[0]*args.stride+args.stride//2), int(max_index[1]*args.stride+args.stride//2)
                    key_pred_position.append((x_pred, y_pred))


            # key_image = draw_circle(key_image, key_pred_position, key_gt_position, colors, args.stride) # prediction
            # cv2.imwrite(os.path.join(args.corres_target, key_image_name), key_image)

            # save the prediction position errors
            draw_and_save(query_image, key_image,
                        query_gt_position, key_gt_box, key_pred_position,
                        save_path=os.path.join('./correspondence/PAX_RAY_spine_query_key/vis', query_image_name.split('.')[0]+'_'+key_image_name.split('.')[0]+'.png'))
            error_list = []
            for i in range(len(key_pred_position)):
                total_points += 1
                x_pred, y_pred = key_pred_position[i]
                x, y, w, h = key_gt_box[i]
                if x <= x_pred <= x + w and y <= y_pred <= y + h:
                    correct_match += 1
                    error = 0
                else:
                    error = point_to_bbox_edge_distance_outside((x_pred, y_pred), (x, y, w, h))
                    # error = math.sqrt((key_pred_position[i][0]-key_gt_position[i][0])**2+(key_pred_position[i][1]-key_gt_position[i][1])**2)
                    non_correct_error.append(error)
                error_list.append(error)
            error_list.insert(0, query_image_name)
            error_list.insert(1, key_image_name)

            writer.writerow(error_list)
            print(error_list)
            file.flush()
            
    file.close()
    print('correct ratio:', correct_match/total_points)
    plt.figure(figsize=(8, 5))

    # 画直方图
    plt.hist(non_correct_error, bins=20, color='skyblue', edgecolor='black', alpha=0.6, label='Histogram')

    # 叠加 KDE 曲线
    sns.kdeplot(non_correct_error, color='red', linewidth=2, label='KDE')

    # 设置图例和标签
    plt.title('Error Distribution')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency / Density')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig('correspondence/spinr_error_consistency_fromIN_150epc.png', dpi=300, bbox_inches='tight')







