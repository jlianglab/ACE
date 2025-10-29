
import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_error(preds, tcoords):
    preds = get_preds(preds)
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    # print(preds.shape) # B,7,2
    # print(tcoords.shape)
    for i in range(N):
        pts_pred, pts_gt = preds[i, ], tcoords[i, ]
        
        rmse[i] = np.sum(np.linalg.norm(pts_pred.cpu() - pts_gt.cpu(), axis=1)) / L

    return rmse


def draw_points_and_save_images(img_tensor, C1, C2, save_dir, image_name):
    """
    Draw red circles for C1 and green 'X' for C2 on images and save them.

    :param img_tensor: The image tensor of shape [B, C, H, W]
    :param C1: The C1 points tensor of shape [B, 7, 2]
    :param C2: The C2 points tensor of shape [B, 7, 2]
    :param save_dir: Directory to save the images
    :return: None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    B, C, H, W = img_tensor.shape
    for b in range(B):
        # Convert the tensor to PIL Image
        img = img_tensor[b].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img.astype('uint8'))

        draw = ImageDraw.Draw(img)
        for point in C1[b]:
            x, y = point.int()
            # Draw red circle
            draw.ellipse((x-3, y-3, x+3, y+3), fill='red')

        for point in C2[b]:
            x, y = point.int()
            # Draw green 'X'
            draw.line((x-3, y-3, x+3, y+3), fill='green', width=3)
            draw.line((x+3, y-3, x-3, y+3), fill='green', width=3)

        # Save the image
        img.save(os.path.join(save_dir, image_name[b]))


def save_images_with_heatmaps(image_tensor, heatmap_tensor, save_dir, image_name):
    """
    Combine heatmap layers, overlay on the images and save the resultant images.

    :param image_tensor: The image tensor of shape [B, 3, H, W]
    :param heatmap_tensor: The heatmap tensor of shape [B, 7, H, W]
    :param save_dir: Directory to save the heatmapped images
    :return: None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    B, _, H, W = image_tensor.shape
    combined_heatmaps = heatmap_tensor.mean(dim=1, keepdim=True)  # Combine heatmaps

    for b in range(B):
        img = image_tensor[b].permute(1, 2, 0).cpu().numpy()
        heatmap = combined_heatmaps[b].squeeze().cpu().numpy()
        heatmap = np.clip(heatmap, 0, 1)  # Ensure heatmap values are within [0, 1]

        # Apply the heatmap to the image
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap, cmap='jet', alpha=0.3, interpolation='nearest')  # Overlay heatmap
        plt.axis('off')

        # Save the image
        plt.savefig(os.path.join(save_dir, image_name[b]), bbox_inches='tight', pad_inches=0)
        plt.close()


def save_images_with_heatmaps_points(image_tensor, heatmap_tensor, C1, C2, save_dir, image_name):
    """
    Combine heatmap layers, overlay on the images, draw predict and target points on the images, and save the resultant images.

    :param image_tensor: The image tensor of shape [B, 3, H, W]
    :param heatmap_tensor: The heatmap tensor of shape [B, 7, H, W]
    :param C1: The C1 points tensor of shape [B, 7, 2]
    :param C2: The C2 points tensor of shape [B, 7, 2]
    :param save_dir: Directory to save the heatmapped images
    :return: None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    B, _, H, W = image_tensor.shape
    combined_heatmaps = heatmap_tensor.mean(dim=1)  # Combine heatmaps

    for b in range(B):
        img = image_tensor[b].permute(1, 2, 0).cpu().numpy()
        heatmap = combined_heatmaps[b].cpu().numpy()
        heatmap = np.clip(heatmap, 0, 1)  # Ensure heatmap values are within [0, 1]

        # Apply the heatmap to the image
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.imshow(heatmap, cmap='jet', alpha=0.3, interpolation='nearest')  # Overlay heatmap
        plt.axis('off')

        # Save the image
        plt.savefig(os.path.join(save_dir, image_name[b]), bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(os.path.join(save_dir, image_name[b]))
        img = img.resize((448,448))
        draw = ImageDraw.Draw(img)
        for point in C1[b]:
            x, y = point.int()
            # Draw red circle
            draw.ellipse((x-3, y-3, x+3, y+3), fill='red')

        for point in C2[b]:
            x, y = point.int()
            # Draw green 'X'
            draw.line((x-3, y-3, x+3, y+3), fill='green', width=3)
            draw.line((x+3, y-3, x-3, y+3), fill='green', width=3)

        # Save the image with heatmap
        img.save(os.path.join(save_dir, image_name[b]))

