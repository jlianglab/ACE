import os
import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
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
import random
import scipy
import scipy.misc


def build_loader_NIHchest_keypoint(config, ddp=False):
    dataset_root = config.DATA.DATA_PATH
    trainpath = config.DATA.TRAIN_LIST
    valpath = config.DATA.VAL_LIST
    testpath = config.DATA.TEST_LIST
    

    train_list = []
    train_label = []
    val_list = []
    val_label = []
    test_list = []
    test_label = []

    if config.MODE == 'train':
        for i in os.listdir(trainpath):
            train_list.append(i)
                
        for i in os.listdir(valpath):
            val_list.append(i)


        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = NIHchest_dataset(dataset_root=dataset_root, landmark_root=trainpath, datalist=train_list, img_transforms=img_train_transforms, config=config)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = NIHchest_dataset(dataset_root=dataset_root, landmark_root=valpath, datalist=val_list, img_transforms=img_val_transforms, config=config)
        

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
                                num_workers=config.DATA.NUM_WORKERS,
                                sampler=sampler_val)
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
        for i in os.listdir(testpath):
            test_list.append(i)
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = NIHchest_dataset(dataset_root=dataset_root, landmark_root=testpath, datalist=test_list, img_transforms=img_test_transforms, config=config)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class NIHchest_dataset(Dataset):
    def __init__(self, dataset_root, landmark_root, datalist, img_transforms, config):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.landmark_root = landmark_root
        self.datalist = datalist
        self.npoints = config.MODEL.NUM_CLASSES
        self.config = config

    def __getitem__(self, index):
        # positions = [2,10,18,34,42,50,21]
        # positions = [2,34,29,24,40,10,30,25,46,16,32,54,53]
        positions = self.config.SELECT_INDEX
        selected_positions = [position - 1 for position in positions]
        # print(selected_positions)
        with open(os.path.join(self.landmark_root, self.datalist[index])) as f:
            content = f.read().strip()
            # Split the content to get image name and coordinates
            image_name, *coords = content.split('#')
            image_name = image_name.split('-')[0] + '.png'
            coords = [[float(coord.split(',')[0]), float(coord.split(',')[1])] for coord in coords if coord != '']
            try:
                selected_coords = [coords[i] for i in selected_positions]
            except Exception as e:
                print(image_name, len(coords))
                print("发生错误:", e)
        tcoords = selected_coords.copy()

        image_init = cv2.imread(os.path.join(self.dataset_root, image_name))
        h,w,_ = image_init.shape
        scale = h*1.0/self.config.DATA.IMG_SIZE
        target = np.zeros((self.npoints, self.config.DATA.IMG_SIZE, self.config.DATA.IMG_SIZE))

        for i in range(self.npoints):
            tcoords[i][0] = tcoords[i][0]/scale
            tcoords[i][1] = tcoords[i][1]/scale
            target[i] = generate_target(target[i], [tcoords[i][0]-1, tcoords[i][1]-1], sigma=8)

        transformed = self.img_transforms(image=image_init)
        image = transformed['image']
        target = torch.Tensor(target)
        tcoords = torch.tensor(tcoords)
        
        transform2 = init_image_transform(self.config.DATA.IMG_SIZE)
        image_init = transform2(image=image_init)['image'] # 1024 --> 448

        return image_init, image.float(), target, tcoords, image_name
    # return image 
    def __len__(self):
        return len(self.datalist)

def img_transforms(mode, config):
    if mode == 'train':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            # albumentations.ShiftScaleRotate(rotate_limit=10),
                                            albumentations.RandomBrightnessContrast(),
                                            ToTensorV2()
                                            ])
        
    elif mode == 'val' or mode == 'test':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            ToTensorV2()
                                            ])


    return data_transforms

def init_image_transform(imgsize):
    data_transforms = albumentations.Compose([
                                            albumentations.Resize(imgsize, imgsize),
                                            ToTensorV2()
                                            ])
    return data_transforms

def generate_target(img, pt, sigma, label_type='Gaussian'): # Gaussian范围 6*sigma+1
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

