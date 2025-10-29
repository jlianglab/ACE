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
import math
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import models.convnext as convnext
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, ViTModel, ViTForImageClassification, AutoConfig
from PIL import Image, ImageDraw



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

def get_embd(model, position, image, device):
    """
    get the embedding of one position
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    patch = crop_and_pad(image, position)
    patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
    # patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_CUBIC)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = Image.fromarray(patch)

    patch = transform(patch).unsqueeze(0).to(device)
    #print(patch.shape)
    with torch.no_grad():
        # Extract features using the model
        features = model.forward_features(patch) # swin:[1,196,768] vit:[1,197,768]
        # features = model(patch) # hugging face
        # features = features.last_hidden_state[:,1:] # hugging face
        # ipdb.set_trace()
        # print(features.shape)
        f1 = features[:,90] # 90 for swin backbone and 91 for vit backbone
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


def interpolation(image, model, ratio, device, embd_dic=None):
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

        embd_a = get_embd(model, (xa,ya), image, device).cpu().numpy()
        embd_b = get_embd(model, (xb,yb), image, device).cpu().numpy()
        embd_c = get_embd(model, (xc,yc), image, device).cpu().numpy()

    predict_embd_c = ratio*embd_a+(1-ratio)*embd_b

    similarity = cosine_similarity(embd_c.reshape(1, -1), predict_embd_c.reshape(1, -1))
    # ipdb.set_trace()
    return similarity[0,0]


def is_in_range(number, start, end):
    return start <= number <= end

def extrapolation(image, model, ratio, device, embd_dic=None):
    """
    embd_dic: embedding dictionary of each image 128*128*1024
    the position of C: C = ratio*A+(1-ratio)B
    """
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
        xa = random.randint(224, 800)
        ya = random.randint(224, 800)

        xb = random.randint(224, 800)
        yb = random.randint(224, 800)

        xc = round(-ratio*xa+(1+ratio)*xb)
        yc = round(-ratio*ya+(1+ratio)*yb)
        while (not is_in_range(xc,224,800)) or (not is_in_range(yc,224,800)) or (xb==xa and yb==ya):
            xa = random.randint(224, 800)
            ya = random.randint(224, 800)

            xb = random.randint(224, 800)
            yb = random.randint(224, 800)

            xc = round(-ratio*xa+(1+ratio)*xb)
            yc = round(-ratio*ya+(1+ratio)*yb)

        embd_a = get_embd(model, (xa,ya), image, device).cpu().numpy()
        embd_b = get_embd(model, (xb,yb), image, device).cpu().numpy()
        embd_c = get_embd(model, (xc,yc), image, device).cpu().numpy()


    predict_embd_c = ratio*embd_a+(1-ratio)*embd_b

    similarity = cosine_similarity(embd_c.reshape(1, -1), predict_embd_c.reshape(1, -1))
    return similarity[0,0]


def triangulation(image, model, device):
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
    embd_P = get_embd(model, (round(P[0]), round(P[1])), image, device).cpu().numpy()
    embd_A = get_embd(model, A, image, device).cpu().numpy()
    embd_B = get_embd(model, B, image, device).cpu().numpy()
    embd_C = get_embd(model, C, image, device).cpu().numpy()
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


def random_crop_and_fill(image):
    width, height = image.size
    
    # Step 1: Randomly crop a square C1 from the image, size is between 0.5 to 0.8 of the original image size
    min_size = int(0.5 * width)
    max_size = int(0.7 * width)
    C1_size = random.randint(min_size, max_size)
    
    # Random top-left corner for C1
    C1_x = random.randint(0, width - C1_size)
    C1_y = random.randint(0, height - C1_size)
    C1 = image.crop((C1_x, C1_y, C1_x + C1_size, C1_y + C1_size))
    
    # Step 2: Randomly crop a square C1_s from C1, size is between 0.2 to 1 times the size of C1
    C1_s_min_size = int(0.3 * C1_size)
    C1_s_max_size = C1_size
    C1_s_size = random.randint(C1_s_min_size, C1_s_max_size)
    
    # Random top-left corner for C1_s inside C1
    C1_s_x = random.randint(0, C1_size - C1_s_size)
    C1_s_y = random.randint(0, C1_size - C1_s_size)
    C1_s = C1.crop((C1_s_x, C1_s_y, C1_s_x + C1_s_size, C1_s_y + C1_s_size))
    
    # Step 3: Create C2 by filling the cropped area in the original image with black
    C2 = image.copy()
    draw = ImageDraw.Draw(C2)
    draw.rectangle([C1_x, C1_y, C1_x + C1_size, C1_y + C1_size], fill=(0,0,0))
    
    # Step 4: Randomly crop a square C2_s from C2, outside of the C1 area
    # We need to ensure C2_s does not overlap with C1
    
    while True:
        # Random size for C2_s
        C2_s_size = random.randint(32, width)  # Choose a reasonable small size
        
        # Random position for C2_s
        C2_s_x = random.randint(0, width - C2_s_size)
        C2_s_y = random.randint(0, height - C2_s_size)
        
        # Check if C2_s is outside of C1
        if not (C1_x < C2_s_x + C2_s_size and C2_s_x < C1_x + C1_size and 
                C1_y < C2_s_y + C2_s_size and C2_s_y < C1_y + C1_size):
            break
    
    C2_s = C2.crop((C2_s_x, C2_s_y, C2_s_x + C2_s_size, C2_s_y + C2_s_size))
    
    return C1, C1_s, C2, C2_s



def dna_test(model, image, device):
    C1, C1_s, C2, C2_s = random_crop_and_fill(image)
    transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])

    C1 = transform(C1).unsqueeze(0).to(device)
    C1_s = transform(C1_s).unsqueeze(0).to(device)
    C2 = transform(C2).unsqueeze(0).to(device)
    C2_s = transform(C2_s).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract features using the model
        C1_features = model.forward_features(C1)[1].mean(dim=1).cpu().numpy()
        C1_s_features = model.forward_features(C1_s)[1].mean(dim=1).cpu().numpy()
        C2_features = model.forward_features(C2)[1].mean(dim=1).cpu().numpy()
        C2_s_features = model.forward_features(C2_s)[1].mean(dim=1).cpu().numpy()

        # C1_features = model(C1).last_hidden_state[:,1:].mean(dim=1).cpu().numpy() # hugging face
        # C1_s_features = model(C1_s).last_hidden_state[:,1:].mean(dim=1).cpu().numpy() # hugging face
        # C2_features = model(C2).last_hidden_state[:,1:].mean(dim=1).cpu().numpy() # hugging face
        # C2_s_features = model(C2_s).last_hidden_state[:,1:].mean(dim=1).cpu().numpy() # hugging face


    cls_C1, cls_C2 = 0,0
    C1_s_C1_sim = cosine_similarity(C1_features, C1_s_features)
    C1_s_C2_sim = cosine_similarity(C2_features, C1_s_features)
    if C1_s_C1_sim>C1_s_C2_sim:
        cls_C1 = 1
    
    C2_s_C1_sim = cosine_similarity(C1_features, C2_s_features)
    C2_s_C2_sim = cosine_similarity(C2_features, C2_s_features)
    if C2_s_C2_sim>C2_s_C1_sim:
        cls_C2 = 1
    return cls_C1, cls_C2



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the properties of interpolation, extrapolation and triangulation.')
    parser.add_argument('--image_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp_overlapglobal/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis/pretrained_weight/fromMIM_extrap_shuffle_compdecomp/checkpoint0075.pth',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/sslgenesis/hierar_compdecomp/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/simmim/ckpt_epoch_100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/comp_decomp/pretrained_weight/hierar_comp_decomp/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/consistency/pretrained_weight/global_local/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/sslgenesis_ablation/patch_shuffling/pretrained_weight/patch_shuffle_student_teacher/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/from_imagenet_matrixcompdecomp/checkpoint0100.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dinocheckpoint0300_swin.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/popar_ablations/POPAR_Swin_448.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/POPAR_PEAC/global_local_consis/last.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    parser.add_argument('--embd_dir', type=str, default='/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/Landmark_embd',  help='key image embeddings saving dictionary.')
    parser.add_argument('--test_list', type=str, default='./Landmark_Annotation', help='key image embeddings saving dictionary.')
    parser.add_argument('--ratio', type=float, default=0.75,  help='ration of OA/AB')
    parser.add_argument('--save_file', type=str, default='./interpolation/extrap_shuffle_compdecomp_extrapolation_0.75.csv', help='the similarity save file')
    parser.add_argument('--device', type=str, default='0',  help='device number')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer(img_size=448,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)

    # model = AutoModel.from_pretrained('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/huggingface/rad-dino',output_hidden_states=True) # load rad-dino pretrained model
    # model = AutoModel.from_pretrained('microsoft/rad-dino') # load rad-dino pretrained model
    # model = convnext.__dict__['convnext_base']() # convnext
    # model_keys = list(model.state_dict().keys())
    # print(model_keys)
    # print(len(model_keys))
    # with open('./model_keys/modelkeys.txt', 'w') as f:
    #     for i in range(len(model_keys)):
    #         f.writelines(model_keys[i]+'\n')

    checkpoint = torch.load(args.model_path, map_location='cpu')

    try:
        checkpoint = checkpoint['student']
        # checkpoint = checkpoint['teacher']
    except:
        checkpoint = checkpoint['model']
        # checkpoint = checkpoint['state_dict']

    
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}

    # with open('./model_keys/ckpt_keys.txt', 'w') as f:
    #         for i in range(len(list(checkpoint_model.keys()))):
    #             f.writelines(list(checkpoint_model.keys())[i]+'\n')
    # ipdb.set_trace()

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

    classify1 = []
    classify2 = []
    for i in os.listdir(args.test_list):
        image_name = i.split('-')[0]
        # embd_dict = torch.load(os.path.join(args.embd_dir, i))
        image = cv2.imread(os.path.join(args.image_dir, image_name+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        cls_C1, cls_C2 = dna_test(model, image, torch.device('cuda'))
        print(cls_C1, cls_C2)
        classify1.append(cls_C1)
        classify2.append(cls_C2)

    merge = classify1+classify2
    print(sum(classify1)/len(classify1))
    print(sum(classify2)/len(classify2))
    print(sum(merge)/len(merge))


