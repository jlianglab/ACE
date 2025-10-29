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
from timm.models.swin_transformer import SwinTransformer
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
                # transforms.Resize((1024,1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
            ])
    return transform


def save_embeddings(model, image, device, stride):
    """
    input the image with size 1024*1024,extract embeddings for each 32*32 patch
    return patch embedding of the whole image, the number is 128*128=1024
    """
    
    w0,h0 = image.shape[:2]
    w_embd, h_embd = w0//stride, h0//stride
    embeddings = torch.zeros((w_embd,h_embd,1024))

    for i in range(w_embd):
        for j in range(h_embd):
            center = (stride/2+stride*i, stride/2+stride*j) # center coordinate for each patch
            # patch = crop_and_pad(image, center, (stride*14,stride*14), stride) # crop size of 448*448 image around the center
            patch = crop_and_pad(image, center, (448,448), 32) # crop size of 448*448 image around the center

            patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = Image.fromarray(patch)

            transforms = transform()

            patch = transforms(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract features using the model
                features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
                # print(features.shape)
                f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone

            embeddings[i,j] = f1
            print(w_embd*i+j)

    embeddings = embeddings.cpu().numpy()

    return embeddings


def draw_circle(image, points, gt_points, colors):
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
    for point, gt_point, color in zip(points, gt_points, colors):
        # x = point[0]*stride+stride//2
        # y = point[1]*stride+stride//2
        x = point[0]
        y = point[1]
        x_gt, y_gt = gt_point
        
        cv2.circle(image, (x_gt,y_gt), size, color_dict[color], -1)  # -1 表示填充圆 gt
        cv2.line(image, (x - cross_size, y - cross_size), (x + cross_size, y + cross_size), color_dict['red'], 8) # prediction
        cv2.line(image, (x - cross_size, y + cross_size), (x + cross_size, y - cross_size), color_dict['red'], 8)
    return image



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--image_dir', type=str, default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0100.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/fundus_from_imagenet_matrixcompdecomp_mlp_overlapglobal/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/dino_official/pretrained_weights/fundus/checkpoint.pth',  help='The root dir of model.')
    parser.add_argument('--query_text_file', type=str, default='/sda1/zhouziyu/ssl/dataset/FIRE/Ground Truth/control_points_A09_1_2.txt',  help='query text file containing query points position.')
    parser.add_argument('--query_image', type=str, default='/sda1/zhouziyu/ssl/dataset/FIRE/Images/A09_1.jpg',  help='query image.')
    parser.add_argument('--key_image', type=str, default='/sda1/zhouziyu/ssl/dataset/FIRE/Images/A09_2.jpg',  help='key image.')
    parser.add_argument('--target_dir', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/correspondence_fundus/predict',  help='key image embeddings saving dictionary.')
    parser.add_argument('--error_file', type=str, default='./correspondence_fundus/error_ACE.csv',  help='save the errors of each predicted landmark.')
    parser.add_argument('--stride', type=str, default=8,  help='compute embedding stride in 1024*1024 size image')
    parser.add_argument('--save_embd', type=bool, default=False,  help='save the key embeddings of the whole image')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    args = parser.parse_args()
    

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), num_classes=3)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
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

    model.to(device)

    colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan', 'tan', 'thistle','honeydew','orchild']

    transf = transform()
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan']
    # positions = [2,34,21,24,10,44,54,53,30] # query positions
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    # positions = [2,10,18,34,42,50,21]
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'magenta','pink', 'cyan', 'tan', 'thistle','honeydew','orchild']
    # positions = [2,34,21,24,10,44,54,53,30,14,29,31,38] # query positions
    # selected_positions = [position - 1 for position in positions]

    
    # # save the error of each predicted position
    # file =  open(args.error_file, mode='w', newline='', encoding='utf-8')
    # landmark_name = ['ImgName', 'Right clavicle', 'Left clavicle','Spinous process','Azygos arch','Right rib5','Left rib6',\
    #                 'Right hemidiaphragm','Left hemidiaphragm','Pulmonary artery','Right rib7','Aortic arch','Left ventricle boarder','Left rib3']
    # writer = csv.writer(file)
    # writer.writerow(landmark_name)  # 写入表头
    # file.flush()
    # for row in data_dicts:
    #     writer.writerow(row)


    key_image = cv2.imread(args.key_image)  
    key_image = cv2.resize(key_image, (1024,1024))               
    key_features = save_embeddings(model, key_image, device, args.stride)
    key_image_name = args.key_image.split('/')[-1]
    # torch.save(key_features, os.path.join(args.embd_dir, key_image_name)+'.npy')



    query_image = cv2.imread(args.query_image)
    w_init,h_init,_ = query_image.shape
    query_image = cv2.resize(query_image, (1024,1024))
    w0,h0,_ = query_image.shape
    w_embd, h_embd = w0//args.stride, h0//args.stride # saving embd number: w_embd*h_embd
    

    ratio = w0*1.0/w_init

    key_pred_position = []
    key_gt_position = []

    f =  open(args.query_text_file, 'r')
    for i in f:
        label = i.strip()
        label = label.split(' ')
        label = [round(float(x) * ratio) for x in label]

        key_gt_position.append((label[2],label[3]))
        # patch = crop_and_pad(query_image, selected_coord, (args.stride*14,args.stride*14), stride=args.stride)
        patch = crop_and_pad(query_image, (label[0], label[1]), (448,448), 32)
        patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        patch = Image.fromarray(patch)

        patch = transf(patch).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features using the model
            features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
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

    key_image = draw_circle(key_image, key_pred_position, key_gt_position, colors) # prediction
    cv2.imwrite(os.path.join(args.target_dir, key_image_name), key_image)

    file =  open(args.error_file, mode='w', newline='', encoding='utf-8')
    landmark_name = ['ImgName', 'Error']
    writer = csv.writer(file)
    writer.writerow(landmark_name)  # 写入表头
    file.flush()


    error_list = []
    for i in range(len(key_pred_position)):
        error = math.sqrt((key_pred_position[i][0]-key_gt_position[i][0])**2+(key_pred_position[i][1]-key_gt_position[i][1])**2)
        error_list.append(error)
    writer.writerow([key_image_name, np.mean(error_list)])
    print(np.mean(error_list))
    file.flush()



    # '''
