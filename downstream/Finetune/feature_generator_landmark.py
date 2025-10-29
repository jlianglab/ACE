import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from einops import rearrange
from torchvision import transforms
import os
import csv
from models.convnext import convnext_base
from models.swin_transformer import SwinTransformer
from models.swin_transformer_v2 import SwinTransformerV2
from models.swin_transformer_ssl import SwinTransformerSSL
from models.upernet import _SwinTransformer
import torch.backends.cudnn as cudnn
# from utils import load_swin_pretrained
import sys
from PIL import Image
import argparse
from models.resnet import resnet50
from models.eva_x import eva_x_base_patch16
import models.convnext as convnext
from transformers import AutoModel

# from transformers import AutoImageProcessor, AutoModel
# from models_vit_medical_mae import vit_base_patch16


class Landmark_Classfication(Dataset):
    def __init__(self, args, pathImageDirectory, pathDatasetFile, positions = [2,34,21,24,10,44,54,53,30]):

        self.args = args
        self.img_list = []
        self.img_label = []
        self.transformSequence_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]),
        ])
        self.image_size = args.img_size
        self.positions = positions
        self.selected_positions = [position - 1 for position in self.positions]

        self.image_coords_dic = {}
        self.pathImageDirectory = pathImageDirectory

        for file_name in os.listdir(pathDatasetFile):
            with open(os.path.join(pathDatasetFile, file_name), 'r') as f:
                # Read the content
                content = f.read().strip()
                # Split the content to get image name and coordinates
                image_name, *coords = content.split('#')
                image_name = image_name.split('-')[0] + '.png'
                # Parse the coordinates
                coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
                # print(len(coords))
                if len(coords) < 54:
                    continue
                else:
                    self.image_coords_dic[image_name] = coords
                    self.img_list.append(image_name)

                # Randomly select 11 coordinates


    def __getitem__(self, index):
        image_name = self.img_list[index]
        coords = self.image_coords_dic[image_name]
        patch_list = None
        
        if args.pretrain_mode in ['PEAC','ACE','Lamps', 'ACEv2', 'ACEv2_swinv2', 'ark']:
            shift = 32
        elif args.pretrain_mode in ['LeADER', 'adamv2','EVA-X','RAD-DINO']: # imgsize 224
            shift = 0
        elif args.pretrain_mode in ['CheSS']:
            shift = 28
        
        for pos in self.selected_positions:
            selected_coord = coords[pos]

            # Read the image
            img = cv2.imread(os.path.join(self.pathImageDirectory, image_name))
            self.img_list.append(image_name)
            patch = self.crop_and_pad(img, selected_coord,(self.image_size,self.image_size), shift)
            patch = cv2.resize(patch, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = Image.fromarray(patch)
            patch = self.transformSequence_img(patch)

            if patch_list is None:
                patch_list = patch.unsqueeze(0)
            else:
                patch_list = torch.cat([patch_list, patch.unsqueeze(0)])
        return image_name, patch_list, torch.arange(0, len(self.selected_positions))
    def crop_and_pad(self, image, center, size=(96, 96), stride=6.86):
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


    def __len__(self):
        return len(self.img_list)


def generate_embedding( source_model,loader, dict_to_save):
    source_model.train(False)

    for idx, (image_name, imageData, labels) in enumerate(loader):
        print("{}/{}".format(idx, len(loader)))
        imageData = imageData.squeeze(0).float().cuda(non_blocking=True)
        labels = labels.squeeze(0)
        image_name = image_name[0]
        with torch.no_grad():
            # embedding = source_model(imageData)

            #### For RAD-DINO ####
            # embedding = source_model(imageData).last_hidden_state[:,1:]
            # embedding = embedding.mean(dim=1)
            #### For RAD-DINO ####

            #### For Medical MAE ####
            # embedding = source_model.forward_features(imageData)
            #### For Medical MAE ####
            if args.embd_dim == 2048: # chess
                imageData = imageData[:,0].unsqueeze(1)
            if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps','EVA-X','CheSS', 'ACEv2', 'ACEv2_swinv2', 'ark']:
                _, features = source_model.forward_features(imageData) # swin:[1,196,768] vit:[1,197,768] resnet50(chess)
            # ipdb.set_trace()
            elif args.pretrain_mode in ['RAD-DINO']:
                features = source_model(imageData) # hugging face
                features = features.last_hidden_state[:,1:] # hugging face
            elif args.pretrain_mode in ['adamv2']:
                features = source_model.extract_features(imageData) # convnext

            if args.pretrain_mode in ['PEAC','ACE','Lamps','EVA-X', 'ACEv2', 'ark']:
                embedding = features[:,90] # 90 for swin backbone and 91 for vit backbone
            elif args.pretrain_mode in ['LeADER', 'adamv2']:
                embedding = features[:,24]
            elif args.pretrain_mode in ['RAD-DINO']:
                embedding = features[:,684] # rad-dino has 1369(37*37) features
            elif args.pretrain_mode in ['CheSS', 'ACEv2_swinv2']:
                embedding = features[:,119] # CheSS has 256(16*16) features


            for (data, label) in zip(embedding,labels):
                data = data.detach().cpu().numpy()
                if dict_to_save.get(image_name) is not None:
                    dict_to_save[image_name].append((data,label))
                else:
                    dict_to_save[image_name] = [(data,label)]


    return dict_to_save


def train_loops(args):
    mode = args.pretrain_mode
    input_size = args.img_size

    

    image_root = args.image_dir

    #image_root = "/data/jliang12/jpang12/dataset/nih_xray14/images/images"

    if args.landmark_num == 54:
        train_dataset = Landmark_Classfication(args, image_root, "./data/Landmark_Annotation", positions=np.arange(1, 55))
    #train_dataset = Landmark_Classfication(image_root, "data/Landmark_Annotation", input_size)
    elif args.landmark_num == 9: # 2,34,21,24,10,44,54,53,30
        train_dataset = Landmark_Classfication(args, image_root, "./data/Landmark_Annotation", positions=[2,10,21,24,30,34,44,53,54])    
    # train_dataset = Landmark_Classfication(image_root, "data/Landmark_Annotation",input_size, positions=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51])

    # train_dataset = Landmark_Classfication(image_root, "data/Landmark_Annotation",input_size, positions= [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 34, 36, 38, 40, 42, 44, 46, 48, 50 ,52])   # 20 back ribs


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)


    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if args.pretrain_mode in ['LeADER','PEAC','ACE','Lamps', 'ACEv2']:
        model = SwinTransformer(img_size=args.img_size,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'ACEv2_swinv2':
        model = SwinTransformerV2(img_size= 512, patch_size=4, window_size=16, embed_dim=128, depths=(2, 2, 18, 2),
                          num_heads=(4, 8, 16, 32), num_classes=3, use_dense_prediction=True)
    elif args.pretrain_mode == 'ark':
        model = SwinTransformerSSL(img_size=args.img_size,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
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

    model.cuda()




    if torch.cuda.is_available():
        #source_model = torch.nn.DataParallel(source_model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.float().cuda()
        cudnn.benchmark = True

    dict_to_save = {}
    dict_to_save = generate_embedding(model, train_loader,dict_to_save)

    fpath = args.embd_dir+f"/{mode}"
    #fpath = "/data/jliang12/jpang12/dataset/nih_xray14/landmark_classfication_{}".format(model)


    if not os.path.exists(fpath):
        os.makedirs(fpath)

    #np.save(os.path.join(fpath,"data_dictionary_w_{}_mlp".format(use_mlp)), dict_to_save)
    np.save(os.path.join(fpath, f"{mode}_{args.landmark_num}landmarks.npy"), dict_to_save)
    #np.save(os.path.join(fpath,"data_dictionary_20_back_ribs_w_{}_mlp".format(use_mlp)), dict_to_save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--image_dir', type=str, default='/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images/',  help='Dictionary of the image file.')
    parser.add_argument('--pretrain_mode', type=str, choices=['ark', 'LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X', 'ACEv2', 'ACEv2_swinv2'], default='ark', help="Choose the pretraining mode")
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_journal/ACE_v2/pretrained_weight/fromIN_unique_multiscale_consis_compdecomp/checkpoint0050.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/adam/Adam-v2_convnext_base.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/nvme1n1/zhouziyu/ACE_swinv2/pretrained_weight/from_imagenet_ACE_swinv2/checkpoint0025.pth',  help='The root dir of model.')
    # parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/eva-x/eva_x_base_patch16_merged520k_mim.pt',  help='The root dir of model.')
    parser.add_argument('--model_path', type=str, default='/mnt/sda/zhouziyu/ssl/pretrained_model/Ark/ark5_teacher_ep200_swinb_projector1376.pth.tar',  help='The root dir of model.')
    
    parser.add_argument('--landmark_num', type=int, default=9,  help='the number of landmark points')
    parser.add_argument('--embd_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_classification_embedding',  help='key image embeddings saving dictionary.')
    parser.add_argument('--img_size', type=int, default=448,  help="the model's pretrain image size")
    parser.add_argument('--embd_dim', type=int, default=768,  help='save the key embeddings of the whole image,768 for eva-x')
    parser.add_argument('--device', type=str, default='6',  help='device number')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    train_loops(args)


