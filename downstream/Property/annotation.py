import os
import cv2
import numpy as np
import matplotlib as plt
import os
import cv2
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import colors


# Generate a list of 54 distinct colors
def generate_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [plt.cm.hsv(hue) for hue in hues]
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
    return colors

# colors = generate_distinct_colors(7)

def draw_circles(image, center, radius, color):
    """Draws two concentric circles centered at the given point."""
    cv2.circle(image, center, radius, color, -1)  # Filled circle for the larger radius
    # cv2.circle(image, center, radius // 4, (255, 255, 255), -1)  # Filled circle for the smaller radius

def convert_colors_to_bgr(cmap):
    """
    将颜色字符串列表转换为 OpenCV 兼容的 BGR 颜色格式。
    """
    bgr_colors = []
    for color_name in cmap:
        # 使用 matplotlib 将颜色字符串转换为 RGB 格式
        rgb_color = colors.to_rgb(color_name)
        # 将 RGB (0-1) 转换为 0-255 并反转为 BGR 格式
        bgr_color = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        bgr_colors.append(bgr_color)
    return bgr_colors


# cmap = ListedColormap(["red", "yellow", "LightCyan", "lime", "magenta", "PaleTurquoise", "orange"])
# cmap = ListedColormap(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown'])
cmap = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink','magenta','cyan','slategray','seashell','skyblue']
cmap_brg = [
    (255, 0, 0),       # blue -> (B, G, R)
    (0, 0, 255),       # red
    (0, 255, 0),       # green
    (0, 255, 255),     # yellow
    (0, 165, 255),     # orange
    (128, 0, 128),     # purple
    (203, 192, 255),   # pink
    (255, 0, 255),     # magenta
    (255, 255, 0),     # cyan
    (112, 128, 144),   # slategray
    (238, 245, 255),   # seashell
    (235, 206, 135)    # skyblue
]
# cmap = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
# Directory with the text files
text_files_dir = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/Landmark_Annotation/'
# Directory with the png files
images_dir = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/'
# images with landmarks
dst_dir = '/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/visualization/'

# positions = [2, 10, 18, 34, 42, 50, 21]
positions = [2,34,10,42,8,40,12,44,14,46,16,48]
positions = [position - 1 for position in positions]
for file_name in os.listdir(text_files_dir):
    # print(file_name)
    # try:
        if file_name == '00000377_004-gt-2-pa.txt': # example
            with open(os.path.join(text_files_dir,file_name), 'r') as f:
                content = f.read().strip()
                image_name, *coords = content.split('#')
                # image_name = '00000377_004.png'
                image_name = image_name.split('-')[0] + '.png'
                image = cv2.imread(images_dir+image_name)
                # print(image.size)
                # coords = [(int(coord.split(',')[0]), int(coord.split(',')[1]), int(coord.split(',')[2])) for coord in coords if coord != '']
                # # print(coords.shape())
                # coords_1 = [coords[i] for i in range(54)]
                coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']

                

                # for i, (x, y, label) in enumerate(coords_1):
                for i, pos in enumerate(positions):
                    # color = colors[i % len(colors)]  # Use a different color for each point
                    selected_coord = coords[pos]
                    color = cmap_brg[i]
                    # color = tuple(int(c * 255) for c in color[:3][::-1])  # Convert to BGR and into 0-255 range
                    # draw_circles(image, (x, y), 20, color)
                    print(selected_coord)
                    draw_circles(image, selected_coord, 20, color)
                    # cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                output_image_name = image_name
                cv2.imwrite(os.path.join(dst_dir,output_image_name), image)
    # except:
    #     print(file_name)
    #     continue
