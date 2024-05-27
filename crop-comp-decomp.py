import torch
import numpy as np
import random
import cv2
from vision_transformer import DINOHead, SimMIM_head, SimMIM_head_SWIN, DenseHead
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleModel(nn.Module):
    def __init__(self, embedding_size=3):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc = nn.Linear(32 * 16 * 16, embedding_size)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print(x.shape)
        x = x.reshape(-1)  # Reshape to vector
        # print(x.shape)
        x = self.fc(x)
        return x


# Get coordinates of crops
def get_coordinates(img, patch_size=32, crop_size=14):
    img_size = img.shape[-1]
    try:
        assert img_size % patch_size == 0
    except AssertionError as e:
        print("Image size should be divisible by patch_size:", e)
        return

    # Now we have img_size and crop_size figured out
    # So now, we make crops
    anchor_points = img_size / patch_size
    # Coordinates for C1
    c1_x = random.randint(0, anchor_points - crop_size * 2)
    c1_y = random.randint(0, anchor_points - crop_size * 2)

    print(c1_x - crop_size + 2, c1_x + crop_size * 2 - 2)
    print(c1_y - crop_size + 2, c1_y + crop_size * 2 - 2)
    # Coordinates for C2
    c2_x = random.randrange(
        max(0, c1_x - crop_size + 2),
        min(anchor_points - crop_size, c1_x + crop_size * 2 - 2),
        2,
    )
    c2_y = random.randrange(
        max(0, c1_y - crop_size + 2),
        min(anchor_points - crop_size, c1_y + crop_size * 2 - 2),
        2,
    )
    return (
        (c1_x, c1_y),
        (c2_x, c2_y),
    )


# Actual Cropping
def crop(img, c1_coords, c2_coords, crop_size):
    c1 = img[
        :,
        c1_coords[0] : c1_coords[0] + crop_size * 2,
        c1_coords[1] : c1_coords[1] + crop_size * 2,
    ]
    c2 = img[
        :,
        c2_coords[0] : c2_coords[0] + crop_size,
        c2_coords[1] : c2_coords[1] + crop_size,
    ]

    return c1, c2


def get_patches(img, patch_size=32, crop_size=14):
    img_size = img.shape[-1]
    try:
        assert img_size % crop_size == 0
    except AssertionError as e:
        print("Image size should be divisible by the crop_size", e)
        return

    try:
        assert img_size % patch_size == 0
    except AssertionError as e:
        print("Image size should be divisible by the patch_size", e)
        return

    if img_size != 448 and crop_size != 14:
        patch_size = img_size / (crop_size * 2)
    elif img_size != 448 and patch_size != 32:
        crop_size = img_size / patch_size

    patches = []
    for i in range(crop_size):
        for j in range(crop_size):
            patches.append(
                img[
                    :,
                    patch_size * i : patch_size * (i + 1),
                    patch_size * j : patch_size * (j + 1),
                ]
            )
    patches = np.array(patches, dtype=np.float32)
    return patches


def get_embeddings(patches, model):
    embeddings = []
    for patch in patches:
        embeddings.append(model(patch).detach().numpy())
    embeddings = np.array(embeddings)
    return embeddings


import numpy as np


def rearrange_embeddings(embeddings, crop_size):
    assert (
        embeddings.shape[0] % crop_size == 0
    ), "Embeddings size mismatch with crop_size"

    # Calculate the multiplier
    multiplier = np.arange(1, embeddings.shape[0] + 1) // (crop_size * 2)

    # Calculate the effective index
    eff_i = np.arange(1, embeddings.shape[0] + 1) % (2 * crop_size)

    # Calculate the final index based on the pattern
    final_index = np.where(
        eff_i % 4 == 1,
        (eff_i + 1) // 2,
        np.where(eff_i % 4 == 2, (eff_i + 2) // 2, crop_size + (eff_i // 2)),
    )

    # Apply adjustments to final index
    final_index -= 1
    final_index += 2 * crop_size * multiplier

    # Convert final index to integers
    final_index = final_index.astype(int)
    print(final_index)
    # Rearrange embeddings using fancy indexing
    rearranged_embeddings = embeddings[final_index]

    return rearranged_embeddings


# Resize
# Get Embeddings from student and teacher
# Reorder the embeddings
# Get embeddings from Comp and decomp head

if __name__ == "__main__":
    patch_size = 32
    crop_size = 14
    img = np.random.rand(3, 1024, 1024).astype(np.float32)
    for _ in range(1):
        c1_coords, c2_coords = get_coordinates(
            img, patch_size=patch_size, crop_size=crop_size
        )
        print(c1_coords, c2_coords, "\n")
        c1, c2 = crop(img, c1_coords, c2_coords, crop_size=crop_size)
        # Resize
        new_size = patch_size * crop_size
        c1 = np.transpose(c1, (1, 2, 0))
        c2 = np.transpose(c2, (1, 2, 0))
        c1 = cv2.resize(c1, (new_size, new_size))
        c2 = cv2.resize(c2, (new_size, new_size))
        c1 = np.transpose(c1, (2, 0, 1))
        c2 = np.transpose(c2, (2, 0, 1))
        # Patches
        c1_patches = get_patches(c1, patch_size, crop_size)
        c2_patches = get_patches(c2, patch_size, crop_size)
        c1_patches = torch.tensor(c1_patches, dtype=torch.float32)
        c2_patches = torch.tensor(c2_patches, dtype=torch.float32)
        # print(c1_patches.shape, c2_patches.shape)
        # Embeddings
        c1_embs = get_embeddings(c1_patches, SimpleModel())
        c2_embs = get_embeddings(c2_patches, SimpleModel())
        # print(c1_embs.shape, c2_embs.shape)
        c1_rearr = rearrange_embeddings(c1_embs, crop_size=crop_size)
