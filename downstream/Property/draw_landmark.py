import cv2
import numpy as np
import os

# 输入数据
data_str = "00000005_003-gt-0-pa.png#402,170,1#296,204,2#388,210,3#262,248,4#370,264,5#246,290,6#352,318,7#234,344,8#354,376,9#212,410,10#328,442,11#208,478,12#322,526,13#188,558,14#326,594,15#176,654,16#336,666,17#194,768,18#350,764,19#224,872,20#512,192,21#498,248,22#494,270,23#444,416,24#384,534,25#392,660,26#422,746,27#492,892,28#590,338,29#618,462,30#700,622,31#744,730,32#630,178,33#752,232,34#634,220,35#776,270,36#648,268,37#794,332,38#674,330,39#808,390,40#684,390,41#804,458,42#690,480,43#798,528,44#700,554,45#800,618,46#704,636,47#802,694,48#688,732,49#812,818,50#692,818,51#788,934,52#740,756,53#244,682,54#"

# 解析数据
parts = data_str.split('#')

img_dir = '/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images'
filename = os.path.join(img_dir, parts[0].split('-')[0]+'.png')  # 获取文件名

# 检查文件是否存在
if not os.path.exists(filename):
    print(f"错误：文件 '{filename}' 不存在！")
    print("请确保文件路径正确或图片在当前目录下。")
    exit()

# 读取图像
image = cv2.imread(filename)

if image is None:
    print(f"错误：无法读取图像文件 '{filename}'")
    exit()

# 创建图像副本用于标注
annotated_image = image.copy()

# 定义字体和大小
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6  # 字体大小
text_thickness = 2  # 字体粗细
circle_thickness = 2  # 圆框粗细
circle_radius = 10  # 圆点半径

# 处理每个点
points = []
for i, part in enumerate(parts[1:]):
    if not part:
        continue
    
    # 解析坐标和序号
    coords = part.split(',')
    if len(coords) < 2:
        continue
    
    try:
        x = int(coords[0])
        y = int(coords[1])
        
        # 有时序号可能在第三位或直接用点的序号
        if len(coords) > 2:
            idx = int(coords[2])
        else:
            idx = i + 1  # 如果没提供序号，使用点的顺序号
        
        points.append((x, y, idx))
        
        # 绘制红色圆点（外部轮廓）
        cv2.circle(annotated_image, (x, y), circle_radius, (0, 0, 255), circle_thickness)
        # 填充红色圆点（内部）
        cv2.circle(annotated_image, (x, y), circle_radius - 2, (0, 0, 255), -1)
        
        # 准备序号文本
        text = str(idx)
        
        # 计算文本尺寸以确定背景框位置
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # 文本框位置（稍微偏移，避免覆盖点）      
        if idx in [13, 17]:
            text_org = (int(x - text_size[0]/2), y - circle_radius - 8)
        else:
            # 文本位置在点的右方（原位置）
            text_org = (x + circle_radius + 3, y + circle_radius // 2)
        
        # 绘制序号背景框（增强可读性）
        cv2.rectangle(annotated_image, 
                     (text_org[0] - 2, text_org[1] - text_size[1] - 2),
                     (text_org[0] + text_size[0] + 2, text_org[1] + 2),
                     (0, 0, 0), -1)  # 黑色背景
        
        # 绘制黄色序号
        cv2.putText(annotated_image, text, text_org, 
                   font, font_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
    
    except (ValueError, IndexError) as e:
        print(f"解析点 '{part}' 时出错: {e}")

# # 添加标题说明图像
# cv2.putText(annotated_image, f"文件: {filename}", (20, 30), 
#            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# cv2.putText(annotated_image, f"共标注点: {len(points)}个", (20, 70), 
#            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# # 显示结果图像
# window_name = f"带标注的图像: {filename}"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 1000, 800)  # 调整窗口大小
# cv2.imshow(window_name, annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存结果图像
output_filename =  "00000005_003_annotated.png"
cv2.imwrite(output_filename, annotated_image)
print(f"标注后的图像已保存为: {output_filename}")