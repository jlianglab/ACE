# code of interpolation, extrapolation and triangulation

import argparse
import random
import csv
import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import ipdb
import cv2
from PIL import Image
from torchvision import transforms
# from timm.models.swin_transformer import SwinTransformer
from models.swin_transformer import SwinTransformer
from models.swin_transformer_v2 import SwinTransformerV2
import math
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import models.convnext as convnext
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, ViTModel, ViTForImageClassification, AutoConfig
from models.resnet import resnet50
from models.eva_x import eva_x_base_patch16



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


def cal_l1(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
    #print(patch.shape)
    with torch.no_grad():
        # Extract features using the model
        # features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
        # _, features = model.forward_features(patch)
        # features = model(patch) # hugging face
        # features = features.last_hidden_state[:,1:] # hugging face
        # ipdb.set_trace()
        # print(features.shape)
        # if args.embd_dim == 2048: # chess
        #         imageData = imageData[:,0].unsqueeze(1)
        if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS', 'ACEv2', 'ACEv2_swinv2', 'ark']:
            _, features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
        # ipdb.set_trace()
        elif args.pretrain_mode in ['RAD-DINO']:
            features = model(patch) # hugging face
            features = features.last_hidden_state#[:,1:] # hugging face
        elif args.pretrain_mode in ['adamv2']:
            features = model.extract_features(patch) # convnext

        # ipdb.set_trace()
        if args.pretrain_mode in ['PEAC','ACE','Lamps', 'ACEv2', 'ark']:
            f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
        elif args.pretrain_mode in ['LeADER']: # , 'adamv2'
            f1 = features[:,24]
        elif args.pretrain_mode in ['EVA-X']:
            f1 = features[:,91] # 91 for vit backbone
        elif args.pretrain_mode in ['adamv2']:
            f1 = torch.mean(features, dim=1)
            # f1 = features[:,24]
        elif args.pretrain_mode in ['RAD-DINO']:
            # f1 = features[:,684] # rad-dino has 1369(37*37) features
            f1 = features[:,0]
        elif args.pretrain_mode in ['CheSS', 'ACEv2_swinv2']:
            f1 = features[:,119] # CheSS has 256(16*16) features
        
        # f1 = features # 90 for convnext

    return f1

# Function to generate a random triangle
def generate_random_triangle():
    while True:
        # A = np.random.randint(224, 800, size=2)
        A = np.random.randint(0, 1023, size=2)
        B = np.random.randint(0, 1023, size=2)
        C = np.random.randint(0, 1023, size=2)
        
        # Check if the points are not collinear
        if np.linalg.det(np.array([B - A, C - A])) != 0:
            return A, B, C




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


def interpolation(image, model, ratio, device, args, embd_dic=None):
    """
    embd_dic: embedding dictionary of each image 128*128*1024
    the position of C: C = ratio*A+(1-ratio)B
    """

    # ipdb.set_trace()
    if embd_dic is not None:
        xa = random.randint(0, 127)
        ya = random.randint(0, 127)

        xb = random.randint(0, 127)
        yb = random.randint(0, 127)
        while xb==xa and yb==ya:
            xb = random.randint(0, 127)
            yb = random.randint(0, 127)

        xc = round(ratio*xa+(1-ratio)*xb)
        yc = round(ratio*ya+(1-ratio)*yb)
    
        embd_a = embd_dic[xa,ya]
        embd_b = embd_dic[xb,yb]
        embd_c = embd_dic[xc,yc]
    else:
        xa = random.randint(224, 800) # no need zero-padding for 1024*1024 image
        ya = random.randint(224, 800)

        xb = random.randint(224, 800)
        yb = random.randint(224, 800)
        while xb==xa and yb==ya:
            xb = random.randint(224, 800)
            yb = random.randint(224, 800)

        xc = round(ratio*xa+(1-ratio)*xb)
        yc = round(ratio*ya+(1-ratio)*yb)

        embd_a = get_embd(model, (xa,ya), image, device, args).cpu().numpy()
        embd_b = get_embd(model, (xb,yb), image, device, args).cpu().numpy()
        embd_c = get_embd(model, (xc,yc), image, device, args).cpu().numpy()

    predict_embd_c = ratio*embd_a+(1-ratio)*embd_b

    similarity = cosine_similarity(embd_c.reshape(1, -1), predict_embd_c.reshape(1, -1))
    # ipdb.set_trace()
    return similarity[0,0]


def is_in_range(number, start, end):
    return start <= number <= end

def extrapolation(image, model, ratio, device, args, embd_dic=None):
    """
    embd_dic: embedding dictionary of each image 128*128*1024
    the position of C: C = ratio*A+(1-ratio)B
    """
    # crop_size = args.img_size
    crop_size = 224
    if embd_dic is not None:
        xa = random.randint(0, 127)
        ya = random.randint(0, 127)

        xb = random.randint(0, 127)
        yb = random.randint(0, 127)

        xc = round(-ratio*xa+(1+ratio)*xb)
        yc = round(-ratio*ya+(1+ratio)*yb)
        while is_in_range(xc,0,127) and is_in_range(yc,0,127):
            xa = random.randint(0, 127)
            ya = random.randint(0, 127)

            xb = random.randint(0, 127)
            yb = random.randint(0, 127)

            xc = round(-ratio*xa+(1+ratio)*xb)
            yc = round(-ratio*ya+(1+ratio)*yb)

        # ipdb.set_trace()
        embd_a = embd_dic[xa,ya]
        embd_b = embd_dic[xb,yb]
        embd_c = embd_dic[xc,yc]

    else:
        xa = random.randint(crop_size//2, 1024-crop_size//2)
        ya = random.randint(crop_size//2, 1024-crop_size//2)

        xb = random.randint(crop_size//2, 1024-crop_size//2)
        yb = random.randint(crop_size//2, 1024-crop_size//2)

        xc = round(-ratio*xa+(1+ratio)*xb)
        yc = round(-ratio*ya+(1+ratio)*yb)
        while (not is_in_range(xc,crop_size//2,1024-crop_size//2)) or (not is_in_range(yc,crop_size//2,1024-crop_size//2)) or (xb==xa and yb==ya):
            xa = random.randint(crop_size//2, 1024-crop_size//2)
            ya = random.randint(crop_size//2, 1024-crop_size//2)

            xb = random.randint(crop_size//2, 1024-crop_size//2)
            yb = random.randint(crop_size//2, 1024-crop_size//2)

            xc = round(-ratio*xa+(1+ratio)*xb)
            yc = round(-ratio*ya+(1+ratio)*yb)

        embd_a = get_embd(model, (xa,ya), image, device, args).cpu().numpy()
        embd_b = get_embd(model, (xb,yb), image, device, args).cpu().numpy()
        embd_c = get_embd(model, (xc,yc), image, device, args).cpu().numpy()


    predict_embd_c = ratio*embd_a+(1-ratio)*embd_b

    similarity = cosine_similarity(embd_c.reshape(1, -1), predict_embd_c.reshape(1, -1))
    return similarity[0,0]


def triangulation(image, model, device, args=None):
    # Generate a random triangle ABC
    A, B, C = generate_random_triangle()
    print(A,B,C)

    # Generate a random point P inside the triangle ABC using barycentric coordinates
    r1, r2 = np.random.rand(2)
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    P = (1 - r1 - r2) * A + r1 * B + r2 * C

    # Find intersection points
    A1 = find_intersection(A,B,C,P)
    B1 = find_intersection(B,A,C,P)
    C1 = find_intersection(C,A,B,P)
    print(A1,B1,C1,P)

    # ipdb.set_trace()
    embd_P = get_embd(model, (round(P[0]), round(P[1])), image, device, args).cpu().numpy()
    embd_A = get_embd(model, A, image, device, args).cpu().numpy()
    embd_B = get_embd(model, B, image, device, args).cpu().numpy()
    embd_C = get_embd(model, C, image, device, args).cpu().numpy()
    # embd_A1 = get_embd(model, (round(A1[0]), round(A1[1])), image, device).cpu().numpy()
    # embd_B1 = get_embd(model, (round(B1[0]), round(B1[1])), image, device).cpu().numpy()
    # embd_C1 = get_embd(model, (round(C1[0]), round(C1[1])), image, device).cpu().numpy()

    t_a1 = (B[0]-A1[0])/(B[0]-C[0]) if (B[0]-C[0])!=0 else (B[1]-A1[1])/(B[1]-C[1])
    print(t_a1)
    embd_A1 = t_a1*embd_C+(1-t_a1)*embd_B
    # using A1,A to predict P
    t = (A[0]-P[0])*1.0/(A[0]-A1[0])
    embd_P1_pred = t*embd_A1+(1-t)*embd_A
    similarity1 = cosine_similarity(embd_P.reshape(1, -1), embd_P1_pred.reshape(1, -1))

    t_b1 = (A[0]-B1[0])/(A[0]-C[0]) if (A[0]-C[0])!=0 else (A[1]-B1[1])/(A[1]-C[1])
    print(t_b1)
    embd_B1 = t_b1*embd_C+(1-t_b1)*embd_A
    # using B1,B to predict P
    t = (B[0]-P[0])*1.0/(B[0]-B1[0])
    embd_P2_pred = t*embd_B1+(1-t)*embd_B
    similarity2 = cosine_similarity(embd_P.reshape(1, -1), embd_P2_pred.reshape(1, -1))

    t_c1 = (A[0]-C1[0])/(A[0]-B[0]) if (A[0]-B[0])!=0 else (A[1]-C1[1])/(A[1]-B[1])
    print(t_c1)
    embd_C1 = t_c1*embd_B+(1-t_c1)*embd_A
    # using C1,C to predict P
    t = (C[0]-P[0])*1.0/(C[0]-C1[0])
    embd_P3_pred = t*embd_C1+(1-t)*embd_C
    similarity3 = cosine_similarity(embd_P.reshape(1, -1), embd_P3_pred.reshape(1, -1))

    return similarity1[0,0], similarity2[0,0], similarity3[0,0]





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the properties of interpolation, extrapolation and triangulation.')
    parser.add_argument('--image_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    parser.add_argument('--pretrain_mode', type=str, choices=['LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X', 'ACEv2', 'ACEv2_swinv2', 'ark'], default='ACEv2', help="Choose the pretraining mode")
    
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/extrap_shuffle_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/sslgenesis/fromscratch_extrap_shuffle_compdecomp_consis/checkpoint.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0125.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/unique_multi_consis_compdecomp/checkpoint0150.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',  help='The root dir of model.')
    
    parser.add_argument('--embd_dir', type=str, default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/Landmark_embd',  help='key image embeddings saving dictionary.')
    parser.add_argument('--test_list', type=str, default='./Landmark_Annotation', help='key image embeddings saving dictionary.')
    parser.add_argument('--ratio', type=float, default=0.5,  help='ration of OA/AB')
    parser.add_argument('--save_file', type=str, default='./interpolation/RAD-DINO_extrapolation_0.5.csv', help='the similarity save file')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    parser.add_argument('--img_size', type=int, default=448, help='image size')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # model = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
    #                     num_heads=(4, 8, 16, 32), num_classes=3)
    if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps', 'ACEv2', 'ark']:
        model = SwinTransformer(img_size=args.img_size,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'ACEv2_swinv2':
        model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'CheSS':
        model = resnet50(num_classes=2)
    elif args.pretrain_mode == 'adamv2':
        model = convnext.__dict__['convnext_base']()
    elif args.pretrain_mode == 'RAD-DINO':
        model = AutoModel.from_pretrained('/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model
    elif args.pretrain_mode == 'EVA-X':
        print('Loading EVA-X model...')
        model = eva_x_base_patch16(pretrained = args.model_path) # eva-x
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        if args.pretrain_mode in ['adamv2', 'ACEv2_swinv2']:
            checkpoint = checkpoint['teacher']
        elif args.pretrain_mode in ['EVA-X']:
            checkpoint = checkpoint['module']
        elif args.pretrain_mode in ['ACEv2']:
            checkpoint = checkpoint['student']
        else:
            checkpoint = checkpoint
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
    model.eval()

    file = open(args.save_file, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['image_name', 'similarity'])  # 写入表头
    file.flush()

    for i in os.listdir(args.test_list):
        image_name = i.split('-')[0]
        # embd_dict = torch.load(os.path.join(args.embd_dir, i))
        image_path = os.path.join(args.image_dir, image_name+'.png')
        image = cv2.imread(image_path)
        # print(f'Processing image: {image_path}')
        # print(image.shape)
        # similarity = interpolation(image, model, args.ratio, device, args)
        # similarity = extrapolation(image, model, args.ratio, device, args)
        similarity1, similarity2, similarity3 = triangulation(image, model, device, args)
        # print(similarity1, similarity2, similarity3)
        print(similarity)

        writer.writerow([image_name, similarity])
        # writer.writerow([image_name, similarity1, similarity2, similarity3])
        # writer.writerow([image_name, similarity])
        file.flush()


