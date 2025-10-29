import os

import albumentations
import cv2
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
# from albumentations.pytorch.transforms import ToTensorV2
# import albumentations.augmentations.transforms as transforms
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image


def build_loader_eyepacs(config, ddp):
    dataset_root = config.DATA.DATA_PATH
    label_root = config.DATA.LABEL_PATH
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST

    train_list = []
    train_label = []
    val_list = []
    val_label = []
    test_list = []
    test_label = []

    data_info = pd.read_csv(label_root, index_col=0) # the first column is used to be the index
        # image_list = np.asarray(data_info.iloc[:,0])
        # label_list = np.asarray(data_info.iloc[:,3])
        
        # self.img_path_list = np.asarray(image_list)
        # self.class_list = np.asarray(label_list)

        # one_hot_encoded = np.zeros((len(self.class_list), 2), dtype=int)
        # one_hot_encoded[np.arange(len(self.class_list)), self.class_list] = 1
        # self.class_list = one_hot_encoded

    if config.MODE == 'train':
        with open(traintxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                train_list.append(i.split('\n')[0])
                label = int(data_info.loc[i.split('.')[0]]['level'])
                train_label.append(label)
        one_hot_encoded = np.zeros((len(train_label), 5), dtype=int)
        one_hot_encoded[np.arange(len(train_label)), train_label] = 1
        train_label = one_hot_encoded

        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                val_list.append(i.split('\n')[0])
                label = int(data_info.loc[i.split('.')[0]]['level'])
                val_label.append(label)
        one_hot_encoded = np.zeros((len(val_label), 5), dtype=int)
        one_hot_encoded[np.arange(len(val_label)), val_label] = 1
        val_label = one_hot_encoded
        


        # train_list, val_list, train_label, val_label = train_test_split(train_list, train_label, test_size=0.1, random_state=24)

        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = eyepacs_dataset(dataset_root=dataset_root, datalist=train_list, labellist=train_label, img_transforms=img_train_transforms)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = eyepacs_dataset(dataset_root=dataset_root, datalist=val_list, labellist=val_label, img_transforms=img_val_transforms)
        

        if ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)

            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    # shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True,
                                    sampler=sampler_train)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)
        else:
            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)


        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                        mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                        prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                        label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        return train_dataset, val_dataset, train_loader, val_loader, mixup_fn
    elif config.MODE == 'test':
        with open(testtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                test_list.append(i.split('\n')[0])
                label = int(data_info.loc[i.split('.')[0]]['level'])
                test_label.append(label)
        one_hot_encoded = np.zeros((len(test_label), 5), dtype=int)
        one_hot_encoded[np.arange(len(test_label)), test_label] = 1
        test_label = one_hot_encoded
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = eyepacs_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_test_transforms)
        test_loader = DataLoader(dataset=test_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class eyepacs_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, img_transforms):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.labellist = labellist

    def __getitem__(self, index):
        # print(os.path.join(self.dataset_root, self.datalist[index]))
        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))
        imageData = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # imageData = Image.open(os.path.join(self.dataset_root, self.datalist[index])).convert('RGB') 
        imageData, mask = crop_image_from_gray(imageData)
        imageData = resize_padding(imageData)
        # imageData = Image.fromarray(imageData)
        image = self.img_transforms(imageData)
        label = torch.FloatTensor(self.labellist[index]) 

        # return image.float(), label, self.datalist[index]
        return image.float(), label
    # return image 
    def __len__(self):
        return len(self.datalist)

def img_transforms(mode, config):
    if mode == 'train':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=7),
                                            
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                                            ])
    elif mode == 'val':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize(config.DATA.CROP_SIZE),
                                            transforms.CenterCrop(config.DATA.IMG_SIZE),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                                            ])
    elif mode == 'test':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize(config.DATA.CROP_SIZE),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            transforms.TenCrop(config.DATA.IMG_SIZE),
                                            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
                                            ])

    return data_transforms

def resize_padding(image,new_size=(1024,1024)):
    h,w = image.shape[:2]  # current shape [height, width]
    r = min(new_size[0] / h, new_size[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    # 计算需要填充的边的像素
    dw, dh = new_size[1] - new_unpad[0], new_size[0] - new_unpad[1]  
    dw /= 2  # 除以2即最终每边填充的像素
    dh /= 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # round(dw,dh - 0.1)直接让小于1的为0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # padding
    # image = np.pad(image,((3,2),(2,3)),'constant',constant_values = (0,0))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) 
    return image


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img, mask