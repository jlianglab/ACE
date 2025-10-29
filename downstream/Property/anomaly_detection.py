import os
import cv2
from collections import defaultdict, deque
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
import ipdb
from models.swin_transformer import SwinTransformerForSimMIM, MaskedAutoencoderViT
from torchvision import transforms
from PIL import Image, ImageOps


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, vitdecoder):
        super(MultiCropWrapper, self).__init__()

        self.backbone = backbone
        self.vitdecoder = vitdecoder


    def forward(self, x, mask=None, MIM=False, mae_manner=False):

        out_g, _middle_features = self.backbone(x, mask, mae_manner)


        B, L, C = _middle_features.shape
  
        return self.vitdecoder(_middle_features)
    

def mask_generator():
    mask = np.zeros((14,14))
    mask[x2:(11+x2), y2:(11+y2)] = 1


if __name__ == "__main__":
    model_path = '/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/extrapolation/extrapolation_feature_alignment_maemanner/checkpoint0100.pth'
    test_file = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/data/data_split/xray14/official/test_official.txt'
    image_root = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = SwinTransformerForSimMIM(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3)
    model = MultiCropWrapper(backbone, MaskedAutoencoderViT())


    checkpoint = torch.load(model_path, map_location='cpu')
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


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
    ])
    model.to(device)

    f = open(test_file, 'r')
    lines = f.readlines()
    for i in lines:
        i = i.strip()
        imgname = i.split(' ')[0]
        img = cv2.imread(os.path.join(image_root, imgname))

        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)
