import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像（灰度或彩色都可）
img_path = '00000001_000.png'  # 请替换成你的图片路径
img = cv2.imread(img_path)

# 检查是否读取成功
if img is None:
    raise FileNotFoundError(f"图像路径错误：{img_path}")

# 如果图像不是 1024x1024，先 resize
img = cv2.resize(img, (1024, 1024))

# 网格间隔
grid_size = 32

# 画网格线（白色线条）
for x in range(0, 1025, grid_size):  # 从0到1024（含）
    cv2.line(img, (x, 0), (x, 1024), color=(255, 255, 255), thickness=2)

for y in range(0, 1025, grid_size):
    cv2.line(img, (0, y), (1024, y), color=(255, 255, 255), thickness=2)

cv2.imwrite('image_with_grid.png', img)