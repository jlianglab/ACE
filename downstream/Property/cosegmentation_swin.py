import torch
import numpy as np
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import faiss
data1 = torch.load('/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_shuffle_5imgs/00000001_000.png.npy')
data2 = torch.load('/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_shuffle_5imgs/00000001_001.png.npy')
data3 = torch.load('/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_shuffle_5imgs/00000002_000.png.npy')
data4 = torch.load('/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_shuffle_5imgs/00000008_001.png.npy')
data5 = torch.load('/mnt/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_embd_shuffle_5imgs/00000143_009.png.npy')
data_list = [data1, data2, data3, data4, data5]

# 将每个数据重塑为二维数组，并存入列表
reshaped_data_list = [data.reshape(-1, data.shape[-1]) for data in data_list]

# 将所有数据垂直堆叠，形成一个大的二维数组，形状为 (16384*5, 1024)
combined_data = np.vstack(reshaped_data_list)

# 使用 KMeans 进行聚类，聚为 8 类
k = 8
d = combined_data.shape[1]  # 特征维度

# 初始化 KMeans 对象
kmeans = faiss.Kmeans(d, k, niter=300, verbose=True, gpu=False)

# 训练 KMeans 模型
kmeans.train(combined_data)

# 获取聚类结果
_, combined_labels = kmeans.index.search(combined_data, 1)
combined_labels = combined_labels.flatten()

# 将聚类标签拆分回每个数据集，并重塑为 (128,128)
labels_list = []
start = 0
for data in data_list:
    num_pixels = data.shape[0] * data.shape[1]
    labels = combined_labels[start:start + num_pixels]
    labels_image = labels.reshape(data.shape[0], data.shape[1])
    labels_list.append(labels_image)
    start += num_pixels

# 创建一个包含 8 种颜色的颜色映射
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'white']
cmap = ListedColormap(colors[:k])

# 绘制并保存每个数据的聚类结果
for idx, labels_image in enumerate(labels_list):
    plt.figure(figsize=(6, 6))
    plt.imshow(labels_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(ticks=range(k))
    plt.title(f'Clustering Result for Data {idx + 1}')
    plt.axis('off')
    plt.savefig(f'faiss_clustering_result_{idx + 1}.png', bbox_inches='tight', pad_inches=0)
    plt.show()