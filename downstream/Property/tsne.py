# 1 landmark

import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from torch.nn import AdaptiveAvgPool2d
from timm.models.swin_transformer import SwinTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from PIL import Image, ImageOps
import torch.nn as nn
import sys
# Directory with the text files
text_files_dir = './Landmark_Annotation'
# Directory with the png files
images_dir = '/sda1/zhouziyu/ssl/dataset/NIHChestX-ray14/images/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Swin Transformer model
# model = SwinTransformer(img_size= 448, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2),
                        #   num_heads=(4, 8, 16, 32), num_classes=3)
#model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
                        mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        drop_rate=0,drop_path_rate=0.1, in_chans = 3, num_classes=1)
def tsne(model_path="./POC_R_T_L.pth",save_name="tsne_plot",flip=0):
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict = modelCheckpoint['model']
    try:
        checkpoint = checkpoint['student']
    except:
        checkpoint = checkpoint['model']
    #checkpoint = checkpoint['student']
    checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
    for key in checkpoint_model.keys():
        #print(key)
        if key in model.state_dict().keys():
            try:
                model.state_dict()[key].copy_(checkpoint_model[key])
            except:
                pass
            print("Copying {} <---- {}".format(key, key))
        else:
            pass
            # print("Key {} is not found".format(key))
    # For normalizing the input image
    normalize = transforms.Normalize(mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    model.cuda()
    output = []
    # flip = 0
    # selected_positions = [2,10,18,34,42,50,21]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    # selected_positions = [1,9,17,33,41,49,20]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    selected_positions = [1]#,99[7,39,11,43,25,45,26,32,21,18,50] #[23,2,28,30]#[2,21,29,25]#[7,39,11,43,25,45,26,32,21,18,50]#[2,12,15,20,21,34,42,52]#[2,10,18,21,24,25,29,34,42,53]#[[7,39,11,43,25,45,26,32,21,18,50]] #random.sample(range(54), 11)
    # Iterate over each file in the directory
    for pos in selected_positions:
        # if pos==99:
        #     flip = 1
        #     pos = 2
        # else:
        #     flip = 0
        for file_name in os.listdir(text_files_dir):
            # Open the file
            try:
                
                with open(os.path.join(text_files_dir, file_name), 'r') as f:
                    # Read the content
                    content = f.read().strip()
                    # Split the content to get image name and coordinates
                    image_name, *coords = content.split('#')
                    # print(image_name)
                    # print(coords)
                    # sys.exit(1)
                    image_name = image_name.split('-')[0] + '.png'
                    # Parse the coordinates
                    coords = [(int(coord.split(',')[0]), int(coord.split(',')[1])) for coord in coords if coord != '']
                    # print(len(coords))
                    # print(coords)
                    # sys.exit(0)
                    if len(coords)<54:
                        continue

                    # Randomly select 11 coordinates

                    selected_coord = coords[pos]

                    # Read the image
                    img = cv2.imread(os.path.join(images_dir, image_name))

                    # For each coordinate, get the 224x224 patch around it
                    feature_vectors = []
                    x,y = selected_coord

                    patch = img[ max(0, y -224):min(img.shape[1], y + 224),
                                max(0, x - 224):min(img.shape[0], x + 224) ]
                    patch = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_CUBIC)
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                    patch = Image.fromarray(patch)
                    if flip==1:
                        print("fliping")
                        patch = ImageOps.mirror(patch)
                        print("fliping finished")


                    # # Create the flipped version of the patch
                    # flipped_patch = ImageOps.mirror(patch)

                    # # Transform the original patch
                    # patch = transform(patch).unsqueeze(0).to(device)

                    # # Transform the flipped patch
                    # flipped_patch = transform(flipped_patch).unsqueeze(0).to(device)

                    patch = transform(patch).unsqueeze(0).to(device)
                    #print(patch.shape)
                    with torch.no_grad():
                        # Extract features using the model
                        features = model.forward_features(patch) #_features
                        #print(features.shape)
                        # Average the output to get a 1x1024 feature vector
                        #avg_pool = AdaptiveAvgPool2d((1,1))
                        features = features.mean(dim=1) #[:,1:]
                        #feature_vectors.append(features.cpu().numpy())

                    # Concatenate these feature vectors
                    #concatenated = np.concatenate(feature_vectors, axis=0)
                    output.append(features.cpu().numpy())
            except:
                #print('pass')
                continue
    #print(len(output)//11)
    a=len(output)//7
    # Convert the output list to a numpy array
    output = np.array(output)


    # The rest of the code remains the same

    # Convert the output list to a numpy array


    # Reshape the output to 2D (11000, 1024)
    
    output = output.reshape(-1, output.shape[-1])
    print(output.shape)
    np.save(save_name+".npy", output)
    #print(output.shape)
    # Use t-SNE to reduce dimensionality to 2
    tsne = TSNE(n_components=2,learning_rate=500,perplexity=50)
    output_tsne = tsne.fit_transform(output)

    x_min, x_max = np.min(output_tsne, 0), np.max(output_tsne, 0)
    output_tsne = output_tsne / (x_max - x_min)

    # Colors for the scatter plot
    # colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown', 'gray','pink', 'cyan', 'magenta']
    colors = ['blue']
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # Create a scatter plot
    for i in range(7):
        if i != 7:
            plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5)
            # Use triangle marker for i = 7

        else:
            plt.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], color=colors[i], alpha=0.5, marker='^')
        # Use default marker for other values of i
        
        # ax.scatter(output_tsne[i*a:(i+1)*a, 0], output_tsne[i*a:(i+1)*a, 1], output_tsne[i*a:(i+1)*a, 2],
        #     color=colors[i])

    plt.savefig(save_name+".png")
#'/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth'
tsne(model_path='/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/contrast_12n_global_inequal_vit.pth',save_name="./images/tsne_plot_localmseonly",flip=0)
#tsne(model_path='/ocean/projects/med230002p/hluo54/local_contrast_pred12N_aug_global_dino_more121_vit/saving_ckpt_CHESTX_new32_meancomponan/checkpoint.pth',save_name="tsne_plot_localdino_vit3",flip=1)
#'/ocean/projects/med230002p/hluo54/local_contrast12N_global_simmim_inverse_noaug/saving_ckpt_CHESTX_new32_meancomponan/checkpoint0300.pth'
# tsne(model_path="./simmim.pth",save_name="tsne_plot4")
# tsne(model_path="./checkpoint0300_baseline.pth",save_name="tsne_plot2")
#tsne(model_path="./checkpoint0300_P_R_T.pth",save_name="tsne_plot7")
