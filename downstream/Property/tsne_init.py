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

# 循环遍历文件夹中的所有文件
for filename in os.listdir('./pics'):
    # 检查文件是否为图像文件
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 使用Pillow打开图像文件并将其转换为NumPy数组
        image_path = os.path.join('./pics', filename)
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = np.array(image)

        # 将图像数组添加到列表中
        image_array_list.append(image_array)

# 将图像数组列表连接为一个NumPy数组
concatenated_images = np.array(image_array_list)
img1_tensor = normalize(torch.from_numpy(concatenated_images).permute(0,3, 1, 2).float() / 255)

# for key in model.state_dict().keys():
#     print(key)

for key in checkpoint_model.keys():
    print(key)
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(checkpoint_model[key])
        # print("Copying {} <---- {}".format(key, key))
    # else:
    #     pass
    #     # print("Key {} is not found".format(key))
with torch.no_grad():
    features = model.forward_features(img1_tensor)
    features = rearrange(features, 'b l c->  (b l) c')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 假设你有一个196x1024的向量，称为“input_matrix”
input_matrix = features

# 使用t-SNE将数据降维到2D
tsne = TSNE(n_components=2, random_state=42)
reduced_matrix = tsne.fit_transform(input_matrix)

n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(reduced_matrix)

# 可视化聚类结果
plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], alpha=0.5, c=labels, cmap='tab20')
plt.colorbar()
plt.title('2D Visualization of Clustering')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()