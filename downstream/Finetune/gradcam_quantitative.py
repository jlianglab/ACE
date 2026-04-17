
import csv
import sys
import cv2
import numpy as np
import torch
import argparse
import json
import os
import csv
from tqdm import tqdm
from typing import List
import pandas as pd
from models import build_model
from torchvision.transforms import Compose, Normalize, ToTensor
from configs.config_ChestXdet import get_config_ChestXdet
from configs.config_NIHchest import get_config_NIHchest
from models.upernet import UperNet_swin

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import ipdb

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/swin/swin_base_patch4_window7_224.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    # easy config modification
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, vit_base, resnet50')
    parser.add_argument('--model_type', type=str, default='swin', help='swin, swinv2')
    parser.add_argument('--batch-size', type=int,default=30, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=500)
    # parser.add_argument('--dataset', type=str,default='ChestXdet', help="JSRT, ChestXdet")
    parser.add_argument('--img_size', type=int, default=448, help='image size of downstream task')
    # parser.add_argument('--pretrain_mode', type=str, default='compose_12N', help='popar_pec_seg, seg_simmim,seg_simmim_global,simmim_global_infonce,simmim_global_barlow')
    # compose_12N
    parser.add_argument('--fold', type=str,default='1', help="10 split of NIHchest dataset")
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/fromMIM_extrap_shuffle_compdecomp_25epcswin_base_448_1/best.pth')
    parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/dino_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/contrast_12n_global_inequalswin_base_448_4/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/ssl_genesis_100epcswin_base_448_2/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/byol_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_pec_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/simmim_global_448_3/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/simmim_global_448_3/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/local_infonce_teacher_inclu_compo_onlypred12N/pretrained_weight/compose_12N/checkpoint0100.pth')

    parser.add_argument('--ratio', type=str, default='100', help='100, 50, 25, 5shot, 10shot')
    parser.add_argument('--seg_part', type=str, default='all', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='train', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--output', default='/sda1/zhouziyu/ssl/downstream_checkpoints/ChestXdet', type=str, metavar='PATH')
    parser.add_argument('--bbox_file', default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/BBox_List_2017.csv', type=str, metavar='PATH')
    parser.add_argument('--test_path', default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images_withBBox/images_in_test/', type=str, metavar='PATH')
    parser.add_argument('--out_path', default='/nvme2/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/dino/', type=str, metavar='PATH')
    parser.add_argument('--test_label', default='./data/data_split/xray14/official/test_official.txt', type=str, metavar='PATH')


    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument('--master_port', type=str, default='12345')

    # args, unparsed = parser.parse_known_args()
    # args = parser.parse_args([])
    args = parser.parse_args()

    config = get_config_NIHchest(args)
    
    # config = get_config(args)

    return args, config



def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 转换为(x_min, y_min, x_max, y_max)
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算并集
    union_area = w1 * h1 + w2 * h2 - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def read_csv(filename, train_size = 512, init_shape = False):
    data = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            # image_name, x, y, w, h = row
            image_name = row[0]
            if len(row) < 5:
                print(f"警告: {image_name}数据不足，跳过")
                data[image_name] = []
                continue
            x, y, w, h = row[1:5]
            if init_shape:
                x = float(x)*train_size/1024
                y = float(y)*train_size/1024
                w = float(w)*train_size/1024
                h = float(h)*train_size/1024
            
            if image_name not in data:
                data[image_name] = []
            data[image_name].append((int(float(x)), int(float(y)), int(float(w)), int(float(h))))
    return data


def read_csv_gt(filename, train_size = 512, init_shape = False):
    data = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            # image_name, x, y, w, h = row
            image_name = row[0]
            if len(row) < 5:
                print(f"警告: {image_name}数据不足，跳过")
                data[image_name] = []
                continue
            x, y, w, h = row[2:6]
            if init_shape:
                x = float(x)*train_size/1024
                y = float(y)*train_size/1024
                w = float(w)*train_size/1024
                h = float(h)*train_size/1024
            
            if image_name not in data:
                data[image_name] = []
            data[image_name].append((int(float(x)), int(float(y)), int(float(w)), int(float(h))))
    return data


def reshape_transform(tensor, height=14, width=14): # swin
        # 去掉cls token
        # result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def reshape_transform_v2(tensor, height=16, width=16):
        # 去掉cls token
        # result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def reshape_transform_vit(tensor, height=14, width=14):
        # 去掉cls token
        # print(tensor.shape) # [1, 197, 768]
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def draw_bbox(img_path, bbox_file, img_size):
    with open(bbox_file, 'r') as f:
        reader = csv.reader(f)
    # print(reader)
        for row in reader:
            if row[0] == 'Image Index':
                continue
            img_name = row[0]
            # print(img_name)
            x1,y1 = int(float(row[2])*img_size/1024), int(float(row[3])*img_size/1024)
            x2,y2 = x1+int(float(row[4])*img_size/1024), y1+int(float(row[5])*img_size/1024)
            image = cv2.imread(os.path.join(img_path, img_name))
            # print(os.path.join(img_path, img_name))
            # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3) # white
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(img_path,img_name), image)


def draw_one_bbox(img_path, boxlist, img_size, color=(255, 255, 255), original_size=False): # draw gt box for one image
    x, y, w, h = boxlist
    x, y, w, h = int(x), int(y), int(w), int(h)

    if original_size:
        x1,y1 = int(float(x)*img_size/1024), int(float(y)*img_size/1024)
        x2,y2 = x1+int(float(w)*img_size/1024), y1+int(float(h)*img_size/1024)
    else:
        x1,y1 = int(x), int(y)
        x2,y2 = x1+int(w), y1+int(h)
    image = cv2.imread(img_path)
    # print(os.path.join(img_path, img_name))
    # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3) # white
    # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imwrite(img_path, image)
    # return image


def draw_bbox2(img_path, save_path, bbox_file):
    with open(bbox_file, 'r') as f:
        reader = csv.reader(f)
    # print(reader)
        for row in reader:
            img_name = row[0]
            if row[0] == 'Image Index':
                continue
            if len(row) < 5:
                print(f"警告: 行数据不足，跳过 {img_name}")
                continue
            
            # print(img_name)
            x1,y1 = int(float(row[1])), int(float(row[2]))
            x2,y2 = x1+int(float(row[3])), y1+int(float(row[4]))
            image = cv2.imread(os.path.join(img_path, img_name))
            # print(os.path.join(img_path, img_name))
            # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.imwrite(os.path.join(save_path,img_name), image)


def get_box(grad_cam):
    # 生成gradcam的bounding box
    # 假设grad_cam是已经生成的Grad-CAM热图，且是灰度图像
    # 标准化热图

    # grad_cam = cv2.normalize(grad_cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # _, thresholded = cv2.threshold(grad_cam, int(255 * 0.9), 255, cv2.THRESH_BINARY)
    
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    thresholded = np.uint8(grad_cam * 255)
    thresholded[thresholded<80]=0
    thresholded[thresholded>=80]=255
    
    # 找到轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 设置面积阈值
    # area_threshold = 100  # 举例，可以根据需要调整
    # # 过滤掉小于阈值的轮廓
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    else:
        max_contour = None  # 或者适当的默认处理

    # 计算每个轮廓的边界框
    if max_contour is not None:
        # bounding_boxes = [cv2.boundingRect(c) for c in max_contour]
        bounding_boxes = cv2.boundingRect(max_contour)
    else:
        bounding_boxes=None
    # print(bounding_boxes)
    # sys.exit(1)
    return bounding_boxes


def generate_bboxes(heatmap, low_thresh=60, high_thresh=180, min_area=10):
    """
    使用轮廓检测的双阈值方法
    """
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 创建二值掩码
    _, low_mask = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    _, high_mask = cv2.threshold(heatmap, high_thresh, 255, cv2.THRESH_BINARY)
    
    # 在低阈值区域内寻找高阈值区域
    final_mask = cv2.bitwise_and(high_mask, low_mask)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有找到轮廓，返回空列表
    if not contours:
        return None
    
    # 找到面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算最大轮廓的面积
    max_area = cv2.contourArea(max_contour)
    
    # 如果最大轮廓面积小于最小面积阈值，返回空列表
    if max_area < min_area:
        return []
    
    # 获取最大轮廓的边界框
    bbox = cv2.boundingRect(max_contour)
    
    return bbox


def bbox_to_mask(bbox: List[float], width: int, height: int) -> np.ndarray:
        """
        Convert bbox [x, y, w, h] to binary mask.
        
        Args:
            bbox: [x, y, w, h] in original image coordinates
            width: Original image width
            height: Original image height
            
        Returns:
            Binary mask [height, width]
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Clamp to image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        x2 = max(0, min(x + w, width))
        y2 = max(0, min(y + h, height))
        
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y:y2, x:x2] = 1
        
        return mask

def is_point_in_box(point, box, new_size=448, original_size=1024):
    """
    判断点是否在矩形框内
    
    参数:
    point: 点的坐标 (x, y)
    box: 矩形框 [x, y, w, h]，其中x,y是左上角坐标，w是宽度，h是高度
    
    返回:
    bool: 如果点在矩形框内（包括边界）返回True，否则返回False
    """
    px, py = point
    bx, by, bw, bh = box
    if original_size:
        bx = int(bx * new_size / original_size)
        by = int(by * new_size / original_size)
        bw = int(bw * new_size / original_size)
        bh = int(bh * new_size / original_size)

    return (px >= bx) and (px <= bx + bw) and (py >= by) and (py <= by + bh)


def main():
    device = torch.device('cuda', 5)
    args, config = parse_option()
    # 加载预训练的 ViT 模型
    model = build_model(config)
    # model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    state_dict = checkpoint['model']

    # with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/resnet50.txt', 'w') as f:
    #     for i in range(len(list(state_dict.keys()))):
    #         f.writelines(list(state_dict.keys())[i]+'\n')
    # # sys.exit(1)

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # model.eval()

    # 判断是否使用 GPU 加速
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)


    
    if config.BACKBONE == 'swin_base':
        if config.MODEL.TYPE == 'swin':
            target_layer = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model,
                    target_layers=target_layer,
                    # use_cuda=device,
                    reshape_transform=reshape_transform)
        elif config.MODEL.TYPE == 'swinv2':
            target_layer = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model,
                target_layers=target_layer,
                # use_cuda=device,
                reshape_transform=reshape_transform_v2)
    elif config.BACKBONE == 'resnet50':
        target_layer = [model.layer4[-1]]
        cam = GradCAM(model=model,
                target_layers=target_layer,
                use_cuda=device)
    elif config.BACKBONE == 'vit_base':
        target_layer = [model.blocks[-1].norm1]
        cam = GradCAM(model=model,
                target_layers=target_layer,
                use_cuda=device,
                    reshape_transform=reshape_transform_vit)
        
    file =  open(os.path.join(args.out_path, 'gradcam_box_test.csv'), mode='w', newline='')
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['Image Index', 'X', 'Y', 'W', 'H', 'confidence', 'class', 'box label', 'gt label'])
    
    file2 = open(os.path.join(args.out_path, 'conf_8class.csv'), mode='w', newline='')
    writer2 = csv.writer(file2)
    
    # get tested images multi-label
    with open(args.test_label, 'r') as f:
        lines = f.readlines()
    test_labels = {}
    for line in lines:
        parts = line.strip().split(' ')
        img_name = parts[0]
        labels = [int(x) for x in parts[1:15]]
        test_labels[img_name] = labels
    
    
    # disease_map = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    disease_map = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    num_classes = len(disease_map)
    
    save_path_red_heatmap_with_gtbox = os.path.join(args.out_path, 'pred_heatmap_with_gtbox/')
    if not os.path.exists(save_path_red_heatmap_with_gtbox):
        os.makedirs(save_path_red_heatmap_with_gtbox)

    for i in disease_map:
        if not os.path.exists(os.path.join(save_path_red_heatmap_with_gtbox, i)):
            os.makedirs(os.path.join(save_path_red_heatmap_with_gtbox, i))
    
    df = pd.read_csv(args.bbox_file)
    image_indices = sorted(df['Image Index'].unique())
    
    pointing_accuracy = {}
    for cls_idx, cls_name in enumerate(disease_map):
        pointing_accuracy[cls_name] = {}
        pointing_accuracy[cls_name]['pointing_correct'] = 0
        pointing_accuracy[cls_name]['total_positive'] = 0
        pointing_accuracy[cls_name]['pointing_accuracy'] = 0.0

    for img_name in image_indices:
        print(f"Processing {img_name}...")
        rgb_img = cv2.imread(os.path.join(args.test_path, img_name))
        w,h,_ = rgb_img.shape

            
        rgb_img = cv2.resize(rgb_img, (args.img_size, args.img_size))
        rgb_img_norm = np.float32(rgb_img)/255
        input_tensor = preprocess_image(rgb_img_norm, mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]) 
        # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 

        if use_cuda:
            input_tensor = input_tensor.to(device)
            
        
        # 计算 grad-cam
        output = model(input_tensor).sigmoid().cpu().detach().numpy()
 
        # save the confidence of 8 classes
        conflist = output[0][:8]
        labels = test_labels[img_name][:8]
        writer2.writerow([img_name, *conflist, *labels])
        
        
        multi_hot = np.zeros(num_classes, dtype=np.uint8)
        img_annotations = df[df['Image Index'] == img_name]
        
        
        for cls_idx, cls_name in enumerate(disease_map):
                    
            # pred information
            pred_conf = output[0][cls_idx]
            
            target_category = [ClassifierOutputTarget(cls_idx)] # 可以指定一个类别，或者使用 None 表示最高概率的类别
            grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
            
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            
            max_val = grayscale_cam.max() # pointing game evaluation
            max_locations = np.where(grayscale_cam == max_val)
            
            # gradcam_box = generate_bboxes(grayscale_cam, low_thresh=80, high_thresh=120, min_area=10)
            gradcam_box = generate_bboxes(grayscale_cam, low_thresh=60, high_thresh=80, min_area=10)
            

            # 将 grad-cam 的输出叠加到原始图像上
            visualization = show_cam_on_image(rgb_img_norm, grayscale_cam)

            # 保存可视化结果
            # cv2.imwrite(os.path.join(save_path_red_heatmap_with_gtbox, cls_name, img_name), visualization)
            
            
            # gt information
            cls_annotations = img_annotations[img_annotations['Finding Label'] == cls_name]
            
            if len(cls_annotations) > 0:
                multi_hot[cls_idx] = 1.0
                # Create segmentation map for this class
                class_mask = np.zeros((h, w), dtype=np.uint8)
                
                for _, row in cls_annotations.iterrows():
                    # Bbox values are in separate columns: Bbox [x,y,w,h],,, 
                    # So we need to get the 4 values from columns 2-5 (0-indexed)
                    bbox_cols = [row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]]
                    
                    # Convert to float, skip if any value is NaN
                    try:
                        bbox = [float(val) for val in bbox_cols if not pd.isna(val)]
                        if len(bbox) != 4:
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    pointing_accuracy[cls_name]['total_positive'] += 1
                    # draw_one_bbox(os.path.join(save_path_red_heatmap_with_gtbox, cls_name, img_name), bbox, args.img_size, original_size=True) # draw gt box

                # if img_name=='00000468_017.png' and cls_name=='Atelectasis':
                #     ipdb.set_trace()
                # Check if any of the max locations fall inside GT mask
                is_correct = False
                for y, x in zip(max_locations[0], max_locations[1]):
                    if is_point_in_box((x,y), bbox, new_size=args.img_size, original_size=1024) and grayscale_cam[y, x] > 0.5: # pixel value threshold
                        is_correct = True
                        break
                
                if is_correct:
                    pointing_accuracy[cls_name]['pointing_correct'] += 1
        
            
            
            if gradcam_box is not None:
                writer.writerow([img_name,*gradcam_box, pred_conf, cls_idx, multi_hot[cls_idx], labels[cls_idx]])
            
                # draw_one_bbox(os.path.join(save_path_red_heatmap_with_gtbox, cls_name, img_name), gradcam_box, args.img_size,  (255, 255, 0)) # draw pred box

            

    # # Calculate pointing accuracy per class
    average_acc = 0
    for cls_idx, cls_name in enumerate(disease_map):
        total_pos = pointing_accuracy[cls_name]['total_positive']
        if total_pos > 0:
            pointing_accuracy[cls_name]['pointing_accuracy'] = pointing_accuracy[cls_name]['pointing_correct'] / total_pos
        else:
            pointing_accuracy[cls_name]['pointing_accuracy'] = 0.0 
        average_acc += pointing_accuracy[cls_name]['pointing_accuracy']    
            
            
        print(f"Class: {cls_name}, Pointing Accuracy: {pointing_accuracy[cls_name]['pointing_accuracy']:.4f} ({pointing_accuracy[cls_name]['pointing_correct']}/{total_pos})")
    print(f"Average Pointing Accuracy: {average_acc/num_classes:.4f}")
        
            
        # # Get annotations for this image
        # img_annotations = df[df['Image Index'] == img_name]
        
        # # Multi-label one-hot
        # multi_hot = torch.zeros(num_classes, dtype=torch.float32)
        
        # # Collect all bboxes per class and create combined segmentation map
        # combined_seg_map = np.zeros((h, w), dtype=np.uint8)
        # class_seg_maps = [np.zeros((h, w), dtype=np.uint8) for _ in range(num_classes)]  # One segmentation map per class

        # for cls_idx, cls_name in enumerate(disease_map):
        #     cls_annotations = img_annotations[img_annotations['Finding Label'] == cls_name]
            
        #     if len(cls_annotations) > 0:
        #         multi_hot[cls_idx] = 1.0
        #         # Create segmentation map for this class
        #         class_mask = np.zeros((h, w), dtype=np.uint8)
                
        #         for _, row in cls_annotations.iterrows():
        #             # Bbox values are in separate columns: Bbox [x,y,w,h],,, 
        #             # So we need to get the 4 values from columns 2-5 (0-indexed)
        #             bbox_cols = [row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5]]
                    
        #             # Convert to float, skip if any value is NaN
        #             try:
        #                 bbox = [float(val) for val in bbox_cols if not pd.isna(val)]
        #                 if len(bbox) != 4:
        #                     continue
        #             except (ValueError, TypeError):
        #                 continue
                    
        #             draw_one_bbox(os.path.join(save_path_red_heatmap_with_gtbox, cls_name, img_name), bbox, args.img_size) # draw gt box
                    # ipdb.set_trace()
                #     bbox_mask = bbox_to_mask(bbox, w, h)
                #     class_mask = np.maximum(class_mask, bbox_mask)
                
                # class_seg_maps[cls_idx] = class_mask  # Store at correct class index
                # combined_seg_map = np.maximum(combined_seg_map, class_mask)
            
            
        # Resize per-class segmentation maps
        # resized_class_masks = []
        # for cls_idx in range(num_classes):
        #     mask = class_seg_maps[cls_idx]
        #     resized_mask = cv2.resize(mask, (args.img_size, args.img_size), 
        #                         order=0, preserve_range=True, anti_aliasing=False)
        #     resized_class_masks.append(resized_mask)
            
    
    # draw_bbox(save_path_red_heatmap_with_gtbox, args.bbox_file, args.img_size) # gt box
    
    # save_path_red_heatmap_with_predbox = os.path.join(args.out_path, 'pred_heatmap_with_predbox/')
    # if not os.path.exists(save_path_red_heatmap_with_predbox):
    #     os.makedirs(save_path_red_heatmap_with_predbox)
    # draw_bbox2(save_path_red_heatmap_with_gtbox, save_path_red_heatmap_with_predbox, os.path.join(args.out_path, 'gradcam_box.csv'))


    


if __name__ == "__main__":
    main()