

import sys
import cv2
import numpy as np
import torch
import argparse
import json
import os
import csv
from tqdm import tqdm
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

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/swinv2_fromIN_unique_multiscale_consis_compdecomp_25epcswinv2_512_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/large_swinv2_fromIN_unique_multiscale_consis_compdecomp_20epcswinv2_512_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/fromIN_unique_multiscale_consis_compdecomp_100epcswin_448_1/best.pth')
    parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/onebranch_3component_24epc_swin_base_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/dino_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_pec_448_1/swin_base_patch4_window7_224/default/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/simmim_global_448_3/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/local_infonce_teacher_inclu_compo_onlypred12N/pretrained_weight/compose_12N/checkpoint0100.pth')

    parser.add_argument('--ratio', type=str, default='100', help='100, 50, 25, 5shot, 10shot')
    parser.add_argument('--seg_part', type=str, default='all', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='train', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--output', default='/sda1/zhouziyu/ssl/downstream_checkpoints/ChestXdet', type=str, metavar='PATH')
    parser.add_argument('--bbox_file', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/BBox_List_2017.csv', type=str, metavar='PATH')
    parser.add_argument('--test_path', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images_withBBox/images_all/', type=str, metavar='PATH')
    parser.add_argument('--out_path', default='/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/Lamps_one_branch/', type=str, metavar='PATH')


    # distributed training
    parser.add_argument("--local_rank", type=int, default=1, help='local rank for DistributedDataParallel')

    parser.add_argument('--master_port', type=str, default='12345')

    # args, unparsed = parser.parse_known_args()
    # args = parser.parse_args([])
    args = parser.parse_args()

    config = get_config_NIHchest(args)
    
    # config = get_config(args)

    return args, config

def reshape_transform(tensor, height=14, width=14):
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

def draw_bbox(img_path, bbox_file, img_size=448):
    with open(bbox_file, 'r') as f:
        reader = csv.reader(f)
    # print(reader)
        for row in reader:
            if row[0] == 'Image Index':
                continue
            img_name = row[0]
            x1,y1 = int(float(row[2])*img_size/1024), int(float(row[3])*img_size/1024)
            x2,y2 = x1+int(float(row[4])*img_size/1024), y1+int(float(row[5])*img_size/1024)
            image = cv2.imread(img_path+img_name)
            # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.imwrite(img_path+img_name, image)



def main():
    device = torch.device('cuda', 5)
    args, config = parse_option()
    # 加载预训练的 ViT 模型
    model = build_model(config)
    # model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')


    state_dict = checkpoint['model']


    # with open('./model_keys/swinv2.txt', 'w') as f:
    #     for i in range(len(list(state_dict.keys()))):
    #         f.writelines(list(state_dict.keys())[i]+'\n')
    # sys.exit(1)

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
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

    


    # 读取输入图像
    testlist = os.listdir(args.test_path)
    
    if os.path.exists(args.out_path) is False:
        os.makedirs(args.out_path)

    for i in testlist:
        rgb_img = cv2.imread(args.test_path+i)
        rgb_img = rgb_img[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (args.img_size, args.img_size))
        rgb_img = np.float32(rgb_img)/255

        # 预处理图像
        input_tensor = preprocess_image(rgb_img, mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]), 

        if use_cuda:
            input_tensor = input_tensor[0].to(device)


        # 计算 grad-cam
        
        target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
        grayscale_cam = grayscale_cam[0, :]

        # 将 grad-cam 的输出叠加到原始图像上
        visualization = show_cam_on_image(rgb_img, grayscale_cam)

        # 保存可视化结果
        # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
        cv2.imwrite(args.out_path+i, visualization)
    
    draw_bbox(args.out_path, args.bbox_file, args.img_size)



if __name__ == "__main__":
    main()
    # args, config = parse_option()
    # # # draw_bbox('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/figures/gradcam/imges_bbox/', args.bbox_file)
    # draw_bbox(args.out_path, args.bbox_file)