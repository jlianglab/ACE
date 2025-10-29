# 参考博客https://blog.csdn.net/wsLJQian/article/details/122112952

import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import glob
import pydicom
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import shutil 

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """
    获取dicom记录的相关信息， 以及encoded_pixels
    """
    data = {}

    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['Rows'] = dicom_data.Rows
    data['Columns'] = dicom_data.Columns
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID

    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values

        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True

        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)

    return data

def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def plot_with_mask_and_bbox(file_path, mask_encoded_list,  rows, columns, figsize=(20,10)):
    pixel_array = pydicom.dcmread(file_path).pixel_array

    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, rows, columns).T for mask_encoded in mask_encoded_list]
    print('mask_decoded:', mask_decoded_list)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,8))

    # print out the xray
    ax[0].imshow(pixel_array, cmap=plt.cm.gray)
    # print the bounding box
    for mask_decoded in mask_decoded_list:
        # print out the annotated area
        ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
        rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
        # 绘制一些特殊的形状和路径，例如矩形
        bbox = patches.Rectangle((cmin, rmin), cmax-cmin, rmax-rmin, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(bbox)   # 将图形添加到图中
    ax[0].set_title('With Mask')
    ax[1].imshow(pixel_array, cmap=plt.cm.gray)
    ax[1].set_title('Raw')

    plt.show()
    fig.savefig('./data/mask_raw.png')
    # plt.pause(15)  # 显示秒数
    # plt.close()


def save_maskImage(file_path, save_dir, mask_encoded_list, rows, columns):
    # file_path: initial .dcm path
    # save_dir: mask save path
    # use the masking function to decode RLE
    mask_decoded_list = [rle2mask(mask_encoded, rows, columns).T for mask_encoded in mask_encoded_list]

    print('mask_decoded_list:', mask_decoded_list)
		
	# 判断是否多个mask信息，简单直接存储，后面再添加处理进行mask合并
    if len(mask_decoded_list)>1:
        n=1
        for mask_decoded in mask_decoded_list:
            cv2.imwrite(save_dir + '/' + os.path.basename(file_path).split('.dcm')[0]+'_'+str(n)+'.png', mask_decoded)
            n+=1
    else:
        for mask_decoded in mask_decoded_list:
            cv2.imwrite(save_dir + '/'+os.path.basename(file_path).replace('.dcm', '.png'), mask_decoded)


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    return cv_img



if __name__ == "__main__":
    raw_dir = '/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/png-mask-train-all/' # 初步mask存储路径
    # # 获取标注信息
    # rles_df = pd.read_csv('/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/train-rle.csv')
    # rles_df.columns = ['ImageId', 'EncodedPixels']
    # print(rles_df.head())

    # # 获取dcm数据信息
    # train_fns = sorted(glob.glob('/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/dicom-images-train/*/*/*.dcm'))
    # # print(train_fns)
    # train_metadata_df = pd.DataFrame()
    # train_metadata_list = []
    # for file_path in tqdm(train_fns):
    #     dicom_data = pydicom.dcmread(file_path)
    #     train_metadata = dicom_to_dict(dicom_data, file_path, rles_df)
    #     train_metadata_list.append(train_metadata)
    # train_metadata_df = pd.DataFrame(train_metadata_list)
    # print(train_metadata_df.head())
    
    # for index, row in train_metadata_df.sample(n=len(train_metadata_list)).iterrows():
    #     file_path = row['file_path']
    #     print(file_path)
    #     rows = row['Rows']
    #     columns = row['Columns']
    #     mask_encoded_list = row['encoded_pixels_list']

    #     save_dir = raw_dir+os.path.basename(file_path).split('.dcm')[0]
    #     os.mkdir(save_dir)

    #     if len(mask_encoded_list) > 0:
    #         if mask_encoded_list[0] != ' -1':
    #             save_maskImage(file_path, save_dir, mask_encoded_list, rows, columns)
    #         else:
    #             mask = np.zeros((columns, rows), dtype=np.uint8)
    #             cv2.imwrite(save_dir +'/'+os.path.basename(file_path).replace('.dcm', '.png'), mask)




    patient_list = os.listdir(raw_dir)
    postProcessed_list = []
    for patient in patient_list:
        mask_path = os.path.join(raw_dir, patient)
        mask_list = os.listdir(mask_path)

        for i, name_m in enumerate(mask_list):
            # print(name_m)
            # Instance_num = name_m.replace('.png', '').split("_")[1]

            mask = cv_imread(mask_path + "/" + name_m)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            w, h = mask.shape

            if patient not in postProcessed_list:
                print('newLoad')
                mask_temp = np.zeros((h, w))
                postProcessed_list.append(patient)
            else:
                print('reLoad')
                mask_temp = cv_imread("/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/png-mask-train/"+name_m.split("_")[0]+".png")

            try:
                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 查找图像轮廓
            except:
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            flag = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if area > 1:     # 去除<3mm的结节（一个坐标点）
                    flag = True
                    cv2.drawContours(mask_temp, [contour], 0, (255, 255, 255), cv2.FILLED)  # 连续绘制，类似于取并集

            if flag:
                cv2.imwrite("/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/png-mask-train/"+name_m.split("_")[0]+".png", mask_temp)
            else:
                shutil.copy(mask_path + "/" + name_m, "/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/png-mask-train/"+name_m)

        print(patient, len(postProcessed_list))


