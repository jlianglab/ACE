import copy
import os
import random
from random import randint, randrange
from typing import Any

import albumentations
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from random import randint, randrange, choices
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def build_loader_compose(config):
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

    with open(traintxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            train_list.append(i.split(' ')[0])
            # label =  i.split(' ')[1:15]
            label = [int(x) for x in i.split(' ')[1:15]]
            train_label.append(label)
    with open(valtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            val_list.append(i.split(' ')[0])
            label = [int(x) for x in i.split(' ')[1:15]]
            val_label.append(label)

    # if config.POPAR_FORM:
    train_list = np.hstack((train_list, val_list))
    train_label = np.vstack((train_label, val_label))

    train_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=train_list, config=config,
                                    img_transforms=img_transforms(config),
                                    popar_transform=build_md_transform(mode='train'))
    print("successfully build train dataset")


    train_loader = DataLoader(dataset=train_dataset, 
                                # sampler=sampler_train,
                                batch_size=config.DATA.BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=config.DATA.NUM_WORKERS,
                                drop_last=True)

    with open(testtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            test_list.append(i.split(' ')[0])
            label = [int(x) for x in i.split(' ')[1:15]]
            test_label.append(label)

    val_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=test_list, config=config,
                                    img_transforms=img_transforms(config=config), 
                                    patch_transforms=patch_transforms(mode='val'),
                                    popar_transform=build_md_transform(mode='val'))
    print("successfully build val dataset")
    # sampler_val = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, shuffle=config.TEST.SHUFFLE
    # )
    val_loader = DataLoader(dataset=val_dataset, 
                            # sampler=sampler_val,
                            batch_size=config.DATA.BATCH_SIZE, 
                            num_workers=config.DATA.NUM_WORKERS)

    return train_dataset, val_dataset, train_loader, val_loader

    

class NIHchest_dataset(Dataset):
    def __init__(self, dataset_root, datalist, config, img_transforms, popar_transform):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.popar_transform = popar_transform
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.image_size = config.DATA.IMG_SIZE
        self.patch_size = config.DATA.PATCH_SIZE

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))

        # global embedding consistency data
        patch, (idx_x1, idx_y1), (idx_x2, idx_y2), (k, l) = self.img_transforms(image)
        sample_index1, sample_index2 = get_index((idx_x1, idx_y1), (idx_x2, idx_y2), (k, l))
        # print(patch.shape)
        patch1 = patch[:,:,0:3]
        patch2 = patch[:,:,3:6]

        return patch1.float(), patch2.float(), sample_index1, sample_index2
    
    def __len__(self):
        return len(self.datalist)


def img_transforms():
    size = 1024
    img_transforms = transforms.Compose([
                        # KeepRatioResize(size),
                        # CenterCrop(size),
                        # GridRandomCrop(size),
                        PatchCrop(size)
    ])
    return img_transforms

def patch_transforms(mode):
    if mode == 'train':
        img_transforms = transforms.Compose([
                            To_Tensor(),
                            transforms.GaussianBlur(kernel_size=5),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation(degrees=7),
                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])
    elif mode == 'val':
        img_transforms = transforms.Compose([
                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])
    return img_transforms

def build_md_transform(mode, dataset = "chexray"):
    transformList_mg = []
    transformList_simple = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])


    if mode=="train":
        transformList_mg.append(Rearrange_and_Norm())
        transformList_mg.append(local_pixel_shuffling)
        transformList_mg.append(nonlinear_transformation)
        transformList_mg.append(transforms.RandomApply([paint], p=0.9))
        transformList_mg.append(torch.from_numpy)
        transformList_mg.append(normalize)
        transformSequence_mg = transforms.Compose(transformList_mg)

        transformList_simple.append(Rearrange_and_Norm())
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)

        return transformSequence_mg, transformSequence_simple
    else:
        transformList_simple.append(Rearrange_and_Norm())
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)
        return transformSequence_simple, transformSequence_simple


def get_index(a, b, c): 
# 输入：a为crop1左上角grid的index，b为patch2左上角grid的index
# 输出：随机挑选出的crop1和crop2对应patch的索引
    """
    get the overlap mask of crop1 and crop2
    input:
      a: crop1's top left corner index, 
      b: crop2's top left corner index,
      c: the h, w rate of crop1, (14*k)*(14*l)
    output: the overlap mask of crop1 and crop2, and the shape of overlap masks are all 14*14
    """
    (idx_x1, idx_y1), (idx_x2, idx_y2), (k, l) = a, b, c

    # 重合部分index范围
    idx_xmin, idx_xmax = max(idx_x1, idx_x2), min((idx_x1+14*l), (idx_x2+14))
    idx_ymin, idx_ymax = max(idx_y1, idx_y2), min((idx_y1+14*k), (idx_y2+14))

    # 找出重合部分在crop1中对应的index list
    overlap_mask_1 = torch.zeros((14,14))
    overlap_mask_1[(idx_ymin-idx_y1)//k : (idx_ymax-idx_y1)//k,(idx_xmin-idx_x1)//l : (idx_xmax-idx_x1)//l] = 1
    overlap_mask_1 = overlap_mask_1.flatten()
    # index1 = torch.nonzero(overlap_mask_1)
    # print(index1)

    overlap_mask_2 = torch.zeros((14,14))
    overlap_mask_2[idx_ymin-idx_y2:idx_ymax-idx_y2,idx_xmin-idx_x2:idx_xmax-idx_x2] = 1
    overlap_mask_2 = overlap_mask_2.flatten()
    # index2 = torch.nonzero(overlap_mask_2)
    # print(index2)
    

    return overlap_mask_1.bool(), overlap_mask_2.bool()


def gaussian_kernel_normalized(size, sigma=1):
    """Generates a (size x size) Gaussian kernel with mean 0 and standard deviation sigma,
    normalized so that the center value is 1."""
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = g / g.sum()  # Normalize to sum to 1
    g = g / g[size//2, size//2]  # Normalize so the center value is 1
    return g


def apply_gaussian_kernel(kernel, point, matrix_size=14):
    half_k = kernel.shape[0] // 2
    matrix = np.ones((matrix_size, matrix_size))
    
    x, y = point
    result_coords = []
    result_weights = []
    
    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            if 0 <= x + i < matrix_size and 0 <= y + j < matrix_size:
                result_coords.append((x + i, y + j))
                result_weights.append(kernel[half_k + i, half_k + j])
    
    return result_coords, result_weights



def upsample(input_bool): # [196,]
    input = input_bool.reshape(14,14).float()
    output_float = F.interpolate(input.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='nearest')
    output_float = output_float.squeeze(0).squeeze(0).reshape(-1) # [784,]
    output_float = torch.nonzero(output_float)
    return output_float

def downsample(input_bool): # [196,]
    input = input_bool.reshape(14,14).float()
    output_float = F.max_pool2d(input.unsqueeze(0).unsqueeze(0), kernel_size=2)
    output_float = output_float.squeeze(0).squeeze(0).reshape(-1) # [49,]
    output_float = torch.nonzero(output_float)
    return output_float



def get_corresponding_indices(overlap_mask_1, overlap_mask_2, idx1_pair, idx2_pair, kl):
    """
    input: 
     overlap_mask_1: overlap index of crop1 (bool, 14*14)
     overlap_mask_2: overlap index of crop2 (bool, 14*14)
     idx1_pair: top left grid index of crop1
     idx2_pair: top left grid index of crop2
     kl: the size rate of crop1
    output: two target matrices of matrix matching, size 196*196

    """
    idx_x1, idx_y1 = idx1_pair
    idx_x2, idx_y2 = idx2_pair
    k, l = kl
    
    # initialize the mapping array as -1
    mapping_array_2_to_1 = np.full(overlap_mask_2.numel(), -1, dtype=int) # shape: (196,)
    mapping_array_1_to_2 = np.full(overlap_mask_1.numel(), -1, dtype=int)

    # compute the offsets
    offset_x = idx_x2 - idx_x1
    offset_y = idx_y2 - idx_y1

    # find True in overlap_mask_2
    true_indices_2 = torch.nonzero(overlap_mask_2).squeeze()

    # initialize the target of matching matrix
    bce_labelsl2s = torch.zeros(196, 196)
    bce_labelss2l = torch.zeros(196, 196)

    if true_indices_2.dim() != 0:
    # 如果是0维张量（标量），则跳过循环
        for idx in true_indices_2:
            # 计算在 14x14 矩阵中的行和列
            row, col = divmod(idx.item(), 14)

            # 计算对应的位置在 overlap_mask_1 中的索引
            corresponding_row = (row + offset_y) // k
            corresponding_col = (col + offset_x) // l
            corresponding_idx = corresponding_row * 14 + corresponding_col

            # 更新映射数组
            mapping_array_2_to_1[idx] = corresponding_idx if overlap_mask_1[corresponding_idx] else -1
            bce_labelss2l[idx,corresponding_idx] = 1 

            # if overlap_mask_1[corresponding_idx]:
            bce_labelsl2s[corresponding_idx,idx] = 1 
    else:
        pass
    
    return  bce_labelss2l, bce_labelsl2s


def get_corresponding_indices_comp(overlap_mask_1, overlap_mask_2, idx1_pair, idx2_pair, kl):
    """
    input: 
     overlap_mask_1: overlap index of crop1 (bool, [196,])
     overlap_mask_2: overlap index of crop2 (bool, [196,])
     idx1_pair: top left grid index of crop1
     idx2_pair: top left grid index of crop2
     kl: the size rate of crop1
    output: two target matrices of matrix matching, size 196*196

    """
    # idx_x1, idx_y1 = idx1_pair
    # idx_x2, idx_y2 = idx2_pair
    # k, l = kl
    
    # initialize the mapping array as -1
    # mapping_array_2_to_1 = np.full(overlap_mask_2.numel(), -1, dtype=int) # shape: (196,)
    # mapping_array_1_to_2 = np.full(overlap_mask_1.numel(), -1, dtype=int)

    # compute the offsets
    # offset_x = idx_x2 - idx_x1
    # offset_y = idx_y2 - idx_y1

    # find True in overlap_mask_2
    true_indices_2 = torch.nonzero(overlap_mask_2).squeeze() # crop2:small crop  overlap patches:4n
    true_indices_1 = torch.nonzero(overlap_mask_1).squeeze() # crop1:big crop  overlap patches:n
    upsample_indices_1 = upsample(overlap_mask_1) # overlap patches:4n
    downsample_indices_2 = downsample(overlap_mask_2) # overlap patches:n


    # initialize the target of matching matrix
    # bce_labelsl2s = torch.zeros(196, 196)
    # bce_labelss2l = torch.zeros(196, 196)
    bce_labelsl2s = torch.zeros(196, 784)
    bce_labelss2l = torch.zeros(196, 49)

    # get 5*5 gaussian kernel
    kernal1 = gaussian_kernel_normalized(size=5)
    kernal2 = gaussian_kernel_normalized(size=3)

    # decomposition
    for i in range(len(true_indices_2)): 
        idx2 = true_indices_2[i] # index in crop2 0~195
        idx1 = upsample_indices_1[i] # index in upsample crop1 0~783

        row, col = divmod(idx1.item(), 28)

        coords, weights = apply_gaussian_kernel(kernal1, (row, col), matrix_size=28) # apply 5*5 gaussion weights to the matrx matching target

        for i in range(len(coords)):
            corresponding_idx = coords[i][0] * 28 + coords[i][1]

            bce_labelsl2s[idx2,corresponding_idx] = weights[i]

    # composition
    for i in range(len(true_indices_1)):
        idx1 = true_indices_1[i] # index in crop1 0~195
        idx2 = downsample_indices_2[i] # index in downsample crop2 0~49

        row, col = divmod(idx2.item(), 7)

        coords, weights = apply_gaussian_kernel(kernal2, (row, col), matrix_size=7) # apply 5*5 gaussion weights to the matrx matching target

        for i in range(len(coords)):
            corresponding_idx = coords[i][0] * 7 + coords[i][1]

            bce_labelss2l[idx1,corresponding_idx] = weights[i]

    
    return  bce_labelss2l, bce_labelsl2s # [196,49], [196,784]


class Rearrange_and_Norm():
    def __call__(self, image):
        # image = cv2.resize(image, (self.size, self.size))
        image = rearrange(image, 'h w c-> c h w')/255
        return image

class To_Tensor():
    def __call__(self, image):
        # if len(np.array(image).shape) == 2:
        #     image = np.array(image)[:,:,None]
        image = rearrange(image, 'h w c-> c h w')/255
        image = torch.from_numpy(image)
        return image

class KeepRatioResize(): # keep ratio and resize: input size -> h*608 / 608*w
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        h,w,c = image.shape
        if h>w:
            new_h = int(self.size*h*1.0/w)
            new_w = self.size
        else:
            new_h = self.size
            new_w = int(self.size*w*1.0/h)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image
    
        
class CenterCrop(): # h*608 / 608*w -> 608*608
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        image = rearrange(image, 'h w c-> c h w')
        image = torch.from_numpy(image)
        image = tf.center_crop(image, [self.size,self.size])
        image = image.numpy()
        image = rearrange(image, 'c h w-> h w c')
        return image

class GridRandomCrop(): # 608*608 -> 576*576 / 304*304 -> 288*288
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        grid_size = int(self.size/19) # the image is cropped into 19*19 grids, and we take 18*18 grids from it
        start_x = randint(0,grid_size-1)
        start_y = randint(0,grid_size-1)
        image = image[start_x:grid_size*18+start_x, start_y:grid_size*18+start_y, :]
        return image

class PatchCrop():
    """
    get grid-wise cropping
    input: the whole image with size 1024*1024, the image is grided to 32*32 patches and each potch size is 32*32
    output: [image,(x1, y1), (x2, y2), (k, l)]
    image: two concated crops which are resized to 448*448, the size of crop2 is fixed with 14*14 patches (imgsize 448*448), and the size of crop1 is (14*k)*(14*l)
    (x1, y1): the grid index of crop1's top left corner
    (x2, y2): the grid index of crop2's top left corner, the size of crop2 is fixed in 14*14 patches
    (k,l): the h, w rate of crop1
    """
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        grid = int(self.size/32)
        # k, l = choices([(1,1), (1,2), (2,1), (2,2)], [3, 1, 1, 1])[0]
        k, l = 1,1
        if k==2 and l==2:
            x1 = randint(0,4)
            y1 = randint(0,4)
            x2 = randrange(x1%2, 19, 2) # randrang不包括尾部的数
            y2 = randrange(y1%2, 19, 2)
        elif k==2 and l==1: # 竖着的矩形
            x1 = randint(0,18)
            y1 = randint(0,4)
            x2 = randrange(max(x1-7,0), min(19, x1+7), 1)
            y2 = randrange(y1%2, 19, 2)
        elif k==1 and l==2: # 横着的矩形
            x1 = randint(0,4)
            y1 = randint(0,18)
            x2 = randrange(x1%2, 19, 2)
            y2 = randrange(max(y1-7,0), min(19, y1+7), 1)
        elif k==1 and l==1:
            x1 = randint(0,18)
            y1 = randint(0,18)
            x2 = randrange(max(x1-14,0), min(19, x1+14), 1)
            y2 = randrange(max(y1-14,0), min(19, y1+14), 1)
        

        patch1 = image[x1*grid:(14*l+x1)*grid, y1*grid:(14*k+y1)*grid, :]
        patch1 = cv2.resize(patch1, (448, 448))
        patch2 = image[x2*grid:(14+x2)*grid, y2*grid:(14+y2)*grid, :]
        image = np.concatenate((patch1, patch2), axis=2)
        return image, (x1, y1), (x2, y2), (k, l)
    
class PatchResize():
    def __init__(self, size):
        self.size = size
    def __call__(self, image, a, b, c):
        patch1 = image[:,:,0:3]
        patch2 = image[:,:,3:6]
        patch1 = cv2.resize(patch1, (448, 448), interpolation=cv2.INTER_AREA)
        patch2 = cv2.resize(patch2, (448, 448), interpolation=cv2.INTER_AREA)
        image = np.concatenate((patch1, patch2), axis=2) # 448*448*6
        return image, a, b, c


def local_pixel_shuffling(x, prob=0.5):

    # if random.random() >= prob:
    #     return x


    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_deps, img_rows, img_cols = x.shape
    num_block = 40
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[:, noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((img_deps, block_noise_size_x,
                                 block_noise_size_y))
        image_temp[:, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    local_shuffling_x = image_temp


    return local_shuffling_x

def nonlinear_transformation(x, prob=0.9):


    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=500)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def paint(img,outpaint_rate = 0.8):


    if random.random() < outpaint_rate:
        return image_out_painting(img)
    else:
        return image_in_painting(img)

def image_in_painting(x):
    img_deps, img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

        noise = np.random.rand(block_noise_size_x,block_noise_size_y) * 1.0

        for i in range(0,img_deps):
            x[i, noise_x:noise_x + block_noise_size_x,noise_y:noise_y + block_noise_size_y] = noise
        cnt -= 1


    return x


def image_out_painting(x):

    img_deps, img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2] ) * 1.0
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    x[:, noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        x[:, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y]
        cnt -= 1


    return x