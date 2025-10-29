# ChestXray14中有984张有BoudingBox的图片，分别提取出所有有bbox的图片以及测试集中有bbox的图片
# 实验发现所有有BBox的图都在测试集中

import csv
import shutil

test_list_file = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/POPAR/data/xray14/official/test_official.txt'
img_path = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/'
img_dst = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images_withBBox/images_all/'
# img_test_dst = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images_withBBox/images_in_test/'
bbox_list_file = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/BBox_List_2017.csv'

test_list = []
with open(test_list_file, encoding='utf-8') as e: # load train list and train label
    list = e.readlines()
    for i in list:
        test_list.append(i.split(' ')[0])

with open(bbox_list_file, 'r') as f:
    reader = csv.reader(f)
    # print(reader)
    for row in reader:
        if row[0] == 'Image Index':
            continue
        img_name = row[0]
        shutil.copy(img_path+img_name, img_dst+img_name)
        # if img_name in test_list:
        #     shutil.copy(img_path+img_name, img_test_dst+img_name)
            


        