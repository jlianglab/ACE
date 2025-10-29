import numpy as np
from sklearn.cluster import KMeans
import cv2
import einops
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from timm.models.swin_transformer import SwinTransformer
import torch
# import PIL.Image
from PIL import Image
from torchvision import transforms
import os
from einops import rearrange
import datashader as ds
import pandas as pd
import colorcet as cc
import matplotlib.pyplot as plt
model = SwinTransformer(img_size=224,patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        num_heads=(4, 8, 16, 32), num_classes=3)

checkpoint = torch.load("./baseline.pth", map_location='cpu')
# state_dict = modelCheckpoint['model']
checkpoint = checkpoint['student']
checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
# del checkpoint_model['head.weight']
# del checkpoint_model['head.bias']
# del checkpoint_model['head.weight']
# del checkpoint_model['head.bias']
checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
# image1 = Image.open('./pic4.jpg').convert("RGB")
#
# # image2 = PIL.Image.open('./pic3.jpg')
# image1 = image1
# # img2 = image2.resize((224, 224))
# img1 = np.asarray(image1)
# img2 = np.asarray(img2)
# Convert the images to tensors and normalize them
normalize = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
])
image_array_list = []

for filename in os.listdir('./clustering'):
    # 检查文件是否为图像文件
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 使用Pillow打开图像文件并将其转换为NumPy数组
        image_path = os.path.join('./clustering', filename)
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = np.array(image)

        # 将图像数组添加到列表中
        image_array_list.append(image_array)

# 将图像数组列表连接为一个NumPy数组
concatenated_images = np.array(image_array_list)
img1_tensor = normalize(torch.from_numpy(concatenated_images).permute(0,3, 1, 2).float() / 255)
# Load the image embeddings
# embeddings is a numpy array of shape (num_images, embedding_size)
# embeddings = np.load("embeddings.npy")
for key in checkpoint_model.keys():
    print(key)
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(checkpoint_model[key])
        # print("Copying {} <---- {}".format(key, key))
    # else:
    #     pass
    #     # print("Key {} is not found".format(key))
with torch.no_grad():
    features = model.forward_features(img1_tensor).mean(dim=1)
    #features = rearrange(features, 'b  c->  (b l) c')

# Perform clustering using K-means
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(features)

# Assign class labels based on cluster centers
cluster_centers = kmeans.cluster_centers_
labels = np.zeros_like(clusters)
for i in range(2):
    print(clusters)
    # mask = (clusters == i)
    # labels[mask] = int(np.mean(cluster_centers[i][:,0]) < np.mean(cluster_centers[1-i][:,0]))

# Print the number of male and female images
num_male = np.sum(labels == 0)
num_female = np.sum(labels == 1)
print(f"Number of male images: {num_male}")
print(f"Number of female images: {num_female}")