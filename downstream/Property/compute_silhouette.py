# compute Silhouette Coefficient 轮廓系数

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# 生成示例数据
# X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
X = np.load('/mnt/nvme1n1/zhouziyu/visualization/images/ark5.npy')

# KMeans 聚类
kmeans = KMeans(n_clusters=9, random_state=0).fit(X)

# 计算整体轮廓系数
score = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", score)

# 计算每个样本的轮廓系数
sample_scores = silhouette_samples(X, kmeans.labels_)
print("Sample Silhouette Scores:", sample_scores)