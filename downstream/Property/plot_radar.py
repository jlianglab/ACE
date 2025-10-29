import numpy as np
import matplotlib.pyplot as plt
import ipdb

def plot_fundus():
    # 定义数据
    labels = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR', 'Average']
    values1 = [93.39, 81.66, 93.79, 97.25, 98.55, 92.92] # ACE
    # values2 = [93.09, 81.47, 93.29, 96.49, 97.78, 92.42] # imagenet
    values2 = [71.02, 57.98, 69.38, 83.63, 90.32, 74.47] # imagenet
    values3 = [71.02, 57.98, 71.64, 82.82, 89.54, 75.26] # lvmmed_vitb_ps16
    values4 = [87.86, 71.68, 89.76, 96.71, 98.03, 88.8] # dino

    # 数据处理
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 使雷达图完整
    values1 += values1[:1]
    values2 += values2[:1]
    values3 += values3[:1]
    values4 += values4[:1]
    angles += angles[:1]

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, values1, color='red', alpha=0.25)


    ax.plot(angles, values2, 's', color='blue', linewidth=1.5, linestyle='--')
    ax.plot(angles, values3, '^', color='green', linewidth=1.5, linestyle='--')
    ax.plot(angles, values4, 'D', color='yellow', linewidth=1.5, linestyle='--')

    ax.plot(angles, values1, 'o', color='red', linewidth=4)

    ax.set_ylim(50, 100)
    # 添加标签
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    # ax.set_xticklabels([])


    # 添加每一圈的值
    # ax.set_yticks([55, 65, 75, 85, 95, 100])
    # ax.set_yticklabels(['1', '2', '3', '4', '5'])

    plt.savefig('./radar/radar_chart.png')

def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(v - min_val) / (max_val - min_val) * 100 for v in values]



def plot_main():
    # 定义数据
    labels = ['ChestX-ray14','RSNA','Shenzhen','CheXpert','JSRT Heart','JSRT Clavicle','VinDr-Rib','SIIM','COVID-QU-Ex','BLUE-3','BLUE-4',\
              'ROUGE-L','BERTScore-F1','DNA-test','Triangulation']
    values1 = [83.4, 74.79, 98.18, 89.53, 95.35, 93.33, 71.13, 81.19, 86.67, 21.36, 15.8, 39.87, 36.62, 88.54, 90.7] # Lamps
    values2 = [80.48, 73.44, 95.61, 88.24, 94.9, 91.74, 66.93, 78.89, 84.78, 20.84, 15.27, 37.6, 35.4, 82.66, 83.8] # DINO
    values3 = [82.78, 74.39, 97.39, 88.81, 95.21, 92.19, 66.4, 78.97, 85.27, 18.53, 13.29, 36.2, 34.08, 58.06, 86.3] # PEAC
    values4 = [81.84, 73.78, 96.97, 88.16, 94.63, 91.77, 63.77, 78.65, 84.58, 21.33, 15.18, 37.3, 36.34, 60.09, 70.6] # POPAR
    values5 = [83.11, 74.33, 97.77, 88.82, 94.86, 92.87, 66.46, 79.64, 86.42, 15.92, 11.47, 33.14, 30.85, 71.4, 73] # RAD-DINO

     # 计算所有数据的最大值，以确保最大值拉到边缘
    max_value = max(np.max(values1), np.max(values2), np.max(values3), np.max(values4), np.max(values5))
    # 数据处理
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 使雷达图完整
    values1 += values1[:1]
    values2 += values2[:1]
    values3 += values3[:1]
    values4 += values4[:1]
    values5 += values5[:1]
    angles += angles[:1]

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, values1, color='red', alpha=0.25)


    ax.plot(angles, values2, 's', color='blue', linewidth=1.5, linestyle='--')
    ax.plot(angles, values3, '^', color='green', linewidth=1.5, linestyle='--')
    ax.plot(angles, values4, 'D', color='yellow', linewidth=1.5, linestyle='--')
    ax.plot(angles, values5, '*', color='slategray', linewidth=1.5, linestyle='--')

    ax.plot(angles, values1, 'o', color='red', linewidth=4)

    # ax.set_ylim(50, 100)
    # 设置坐标轴的范围，确保最大值对应外边缘
    ax.set_ylim(0, max_value)
    # 添加标签
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(labels)
    ax.set_xticklabels([])


    # 添加每一圈的值
    # ax.set_yticks([55, 65, 75, 85, 95, 100])
    # ax.set_yticklabels(['1', '2', '3', '4', '5'])

    plt.savefig('./radar/radar_main.png')



if __name__ == '__main__':
    plot_main()