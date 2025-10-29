# load label of PAX-RAY++ dataset
import pandas as pd
import json

paxray_test = '/mnt/sda/zhouziyu/ssl/datasets/ChestXray/PAX_RAY/paxray_test.json'
with open(paxray_test, 'r') as f:
    data = json.load(f)

data_image = data['images']
data_categories = data['categories']
data_anno = data['annotations']
print(data_image[:10])
print(data_categories[:10])
print(data_anno[:10])