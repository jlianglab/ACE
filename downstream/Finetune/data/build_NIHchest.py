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


def build_loader_NIHchest(config, ddp):
    dataset_root = config.DATA.DATA_PATH
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST

    train_list = []
    train_label = []
    val_list = []
    val_label = []
    test_list = []
    test_label = []

    if config.MODE == 'train':
        with open(traintxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                train_list.append(i.split(' ')[0])
                # label = i.split(' ')[1:15]
                label = [int(x) for x in i.split(' ')[1:15]]
                train_label.append(label)
        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                val_list.append(i.split(' ')[0])
                label = [int(x) for x in i.split(' ')[1:15]]
                val_label.append(label)

        if config.POPAR_FORM:
            train_list = np.hstack((train_list, val_list))
            train_label = np.vstack((train_label, val_label))

        # 10折交叉验证
        rkf = KFold(n_splits=10, shuffle=False)
        for fold, (train_index, val_index) in enumerate(rkf.split(train_list)): # rkf.split返回的是train和val的index
            locals()['train_list'+str(fold)] = []
            locals()['val_list'+str(fold)] = []
            locals()['train_label'+str(fold)] = []
            locals()['val_label'+str(fold)] = []
            for i in train_index:
                locals()['train_list'+str(fold)].append(train_list[i])
                locals()['train_label'+str(fold)].append(train_label[i])
            for i in val_index:
                locals()['val_list'+str(fold)].append(train_list[i])
                locals()['val_label'+str(fold)].append(train_label[i])

        train_list = locals()['train_list'+config.DATA.FOLD] 
        val_list = locals()['val_list'+config.DATA.FOLD] 
        train_label = locals()['train_label'+config.DATA.FOLD] 
        val_label = locals()['val_label'+config.DATA.FOLD] 

        # train_list, val_list, train_label, val_label = train_test_split(train_list, train_label, test_size=0.1, random_state=24)

        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=train_list, labellist=train_label, img_transforms=img_train_transforms)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=val_list, labellist=val_label, img_transforms=img_val_transforms)
        

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
                                # sampler=sampler_val,
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
                test_list.append(i.split(' ')[0])
                label = [int(x) for x in i.split(' ')[1:15]]
                test_label.append(label)
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_test_transforms)
        test_loader = DataLoader(dataset=test_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class NIHchest_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, img_transforms):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.labellist = labellist

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))
        label = torch.FloatTensor(self.labellist[index]) 
        image = self.img_transforms(image)

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

# def img_transforms(mode, config):
# if mode == 'train':
# data_transforms = albumentations.Compose([
# albumentations.Normalize(mean=(0.5056, 0.5056, 0.5056), std=(0.252, 0.252, 0.252)), 
# albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), 
# albumentations.HorizontalFlip(p=0.25),
# # albumentations.VerticalFlip(p=0.25),
# albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.25), # 随机位移+旋转+缩放，图像尺度不变，padding值为0
# albumentations.OneOf([
# albumentations.Blur(blur_limit=5),
# albumentations.GaussianBlur(blur_limit=5),
# albumentations.MedianBlur(blur_limit=5),
# albumentations.MotionBlur(blur_limit=5)
# ], p=0.25),
# ToTensorV2() 
# ])
# elif mode == 'val' or mode == 'test':
# data_transforms = albumentations.Compose([
# albumentations.Normalize(mean=(0.5056, 0.5056, 0.5056), std=(0.252, 0.252, 0.252)), 
# albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), 
# ToTensorV2() 
# ])

# return data_transforms