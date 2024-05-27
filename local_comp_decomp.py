import torch
import numpy as np
import random
import cv2
import torch.nn as nn
import torch.nn.functional as F
import ipdb

# Get coordinates of crops
def get_coordinates(img, patch_size=32, crop_size=14):
    # Input : image, patch_size, crop_size
    # Output : Coordinates of C1 and C2 with granularity of patch_size
    # ipdb.set_trace()
    img_size = img.shape[1]
    try:
        assert img_size % patch_size == 0
    except AssertionError as e:
        print("Image size should be divisible by patch_size:", e)
        return

    # Now we have img_size and crop_size figured out
    # So now, we make crops
    anchor_points = img_size / patch_size

    # Coordinates for C1
    # Since C1 has size (crop_size*2, crop_size*2)
    # its top coordinates can range from [0 : anchor_points - crop_size*2]
    c1_x = random.randint(0, int(anchor_points - crop_size * 2))
    c1_y = random.randint(0, int(anchor_points - crop_size * 2))
    # C1's end coordinates would be c1_x + (crop_size * 2), c1_y + (crop_size * 2)

    # Coordinates for C2
    # Since C2 has size (crop_size, crop_size), its top coordinates can range from
    # atleast [C1's top coordinates - crop_size + 2 (since we want to guarantee some overlap) :
    # atmost C1's end coordinates - 2 (since we want to guarantee some overlap)]
    c2_x = random.randrange(
        max(0, int(c1_x - crop_size + 2)),
        min(int(anchor_points - crop_size), int(c1_x + crop_size * 2 - 2)),
        2,
    )
    c2_y = random.randrange(
        max(0, int(c1_y - crop_size + 2)),
        min(int(anchor_points - crop_size), int(c1_y + crop_size * 2 - 2)),
        2,
    )
    # print(c1_x, c1_y, c2_x, c2_y)
    return (
        (c1_x, c1_y),
        (c2_x, c2_y),
    )


# Actual Cropping
def crop(img, c1_coords, c2_coords, crop_size, patch_size):
    # Input: Coordinates of crops with granularity of patch_size
    # Output: Cropped 
    c1 = img[
        c1_coords[0] * patch_size: (c1_coords[0] + crop_size * 2) * patch_size,
        c1_coords[1] * patch_size: (c1_coords[1] + crop_size * 2) * patch_size,
        :,
    ]
    c2 = img[
        c2_coords[0] * patch_size: (c2_coords[0] + crop_size) * patch_size,
        c2_coords[1] * patch_size: (c2_coords[1] + crop_size) * patch_size,
        :,
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

    patches = [] #Array of len = crop_size*crop_size
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

def rearrange_embeddings(ip, crop_size=14):
    # Input: ip : (batch_size, 196, 1) or (batch_size, 196)
    # Output: rearranged_input : (batch_size, 196, 1) or (batch_size, 196)
    '''
    Rearrange embeddings for a batch of inputs.
    '''
    if ip.dim() == 3 and ip.shape[-1] == 1:
        ip = ip.squeeze(-1)  # Convert shape from (batch_size, 196, 1) to (batch_size, 196)

    batch_size = ip.shape[0]
    input_shape = ip.size()
    last_dim = input_shape[-1]

    def apply_logic(embeddings):
        assert (
            embeddings.shape[0] % crop_size == 0
        ), f"Embeddings length {embeddings.shape[0]} mismatch with crop_size {crop_size}"
        # Calculate the multiplier
        multiplier = np.arange(embeddings.shape[0]) // (crop_size * 2)

        # Calculate the effective index
        eff_i = np.arange(1, embeddings.shape[0] + 1) % (2 * crop_size)
        eff_i[eff_i == 0] = 2 * crop_size

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

        # Rearrange embeddings using fancy indexing
        rearranged_embeddings = embeddings[final_index]
        return rearranged_embeddings

    # Apply the rearrangement logic to each sample in the batch
    rearranged_input = torch.stack([torch.tensor(apply_logic(sample)) for sample in ip])

    # Restore the original shape if necessary
    if last_dim == 1:
        rearranged_input = rearranged_input.unsqueeze(-1)

    return rearranged_input

    # Apply the rearrangement logic to each item in the batch
    results = torch.cat([apply_logic(input_tensor_reshaped[i].squeeze(0)) for i in range(batch_size)], dim=0)
    
    # Reshape back to original shape with batch dimension
    reshaped = results.view(batch_size, *input_shape[1:])
    return reshaped

# Function to get Composition labels
def get_comp_gt(c2_overlap_gt_batch: torch.Tensor, crop_size=14):
    # Input : c2_overlap_gt_batch : (batch_size, 196, 1)
    # Output : comp_embeddings_batch : (batch_size, 49, 1)
    batch_size = c2_overlap_gt_batch.shape[0]
    comp_embeddings_batch = torch.empty((batch_size, 49, 1))
    
    for batch_idx in range(batch_size):
        c2_overlap_gt = c2_overlap_gt_batch[batch_idx]
        comp_embeddings = torch.empty((0,))
        for i in range(0, c2_overlap_gt.shape[0], crop_size * 2):
            for j in range(0, crop_size, 2):
                if c2_overlap_gt[i + j]:
                    comp_embeddings = torch.cat((comp_embeddings, torch.tensor([1])), dim=0)
                else:
                    comp_embeddings = torch.cat((comp_embeddings, torch.tensor([0])), dim=0)
        comp_embeddings_batch[batch_idx] = comp_embeddings.unsqueeze(-1)
    
    return comp_embeddings_batch

# Function to get Decomposition labels
def get_decomp_gt(c1_overlap_gt_batch: torch.Tensor):
    # Input : c1_overlap_gt_batch : (batch_size, 196, 1)
    # Output : c1_decomp_embeddings_batch : (batch_size, 784, 1)
    batch_size = c1_overlap_gt_batch.shape[0]
    c1_decomp_embeddings_batch = torch.zeros((batch_size, 784, 1))
    
    for batch_idx in range(batch_size):
        c1_overlap_gt = c1_overlap_gt_batch[batch_idx]
        c1_decomp_embeddings = torch.zeros((c1_overlap_gt.shape[0] * 4, 1))
        for i, item in enumerate(c1_overlap_gt):
            c1_decomp_embeddings[i * 4 : i * 4 + 4] = item
        c1_decomp_embeddings_batch[batch_idx] = c1_decomp_embeddings
    
    return c1_decomp_embeddings_batch

def get_comp_decomp_barlow_labels(c1_decomp_gt : torch.Tensor,
                                    c2_comp_gt : torch.Tensor,
                                    c1_locations_batch : torch.Tensor,
                                    c2_locations_batch : torch.Tensor,
                                    crop_size=14):
    # Input : c1_decomp_gt : (784, 1) , c2_comp_gt : (49, 1), c1_locations : (196, 1), c2_locations : (196, 1)\
    # Output : comp_barlow_labels : (196, 49) , decomp_barlow_labels : (196, 784)

    def process_comp_barlow_labels_batch(c2_comp_gt_batch: torch.Tensor, c1_locations_batch: torch.Tensor, crop_size=14):
        # Input : c2_comp_gt_batch : (batch_size, 49, 1)
        # Input : c1_locations_batch : (batch_size, 196, 1)
        # Output : comp_barlow_labels_batch : (batch_size, 196, 49)
        batch_size = c2_comp_gt_batch.shape[0]
        comp_barlow_labels_batch = torch.zeros((batch_size, 196, 49))
        
        for batch_idx in range(batch_size):
            c2_comp_gt = c2_comp_gt_batch[batch_idx]
            c1_locations = c1_locations_batch[batch_idx]
            
            comp_barlow_labels = torch.zeros((c1_locations.shape[0], c2_comp_gt.shape[0]))
            idx = torch.argmax(c1_locations).item()
            # Top indices of C1
            c1_row_min = idx // crop_size
            c1_col_min = idx % crop_size
            # Top indices of C2 composition
            c2_comp_row_min = torch.argmax(c2_comp_gt).item() // (crop_size // 2)
            c2_comp_col_min = torch.argmax(c2_comp_gt).item() % (crop_size // 2)
            # Number of steps
            num_row = torch.sum(torch.any(c1_locations.reshape(14, 14), dim=1)).item()
            num_col = torch.sum(c1_locations.reshape(14, 14)[c1_row_min]).item()
        
            for r in range(num_row):
                for c in range(num_col):
                    c1_idx = (c1_row_min + r) * crop_size + (c1_col_min + c)
                    c2_comp_idx = int((c2_comp_row_min + r) * (crop_size // 2) + (c2_comp_col_min + c))
                    comp_barlow_labels[c1_idx][c2_comp_idx] = True
                    
            comp_barlow_labels_batch[batch_idx] = comp_barlow_labels
        
        return comp_barlow_labels_batch

    
    def process_decomp_barlow_labels_batch(c1_decomp_gt_batch, c2_locations_batch, decomposition_factor=4, crop_size=14):
        # Input : c1_decomp_gt_batch : (batch_size, 784, 1)
        # Input : c2_locations_batch : (batch_size, 196, 1)
        # Output : decomp_barlow_labels_batch : (batch_size, 196, 784)
        '''
        We will see which of the 196 embeddings in C2 ground truths correspond to the positive embeddings in decomposition embeddings
        '''
        batch_size = c1_decomp_gt_batch.shape[0]
        decomp_barlow_labels_batch = torch.zeros((batch_size, c2_locations_batch.shape[1], c1_decomp_gt_batch.shape[1]))

        c2_locations_rearr_batch = rearrange_embeddings(c2_locations_batch)

        for i in range(batch_size):
            c1_decomp_gt = c1_decomp_gt_batch[i]
            c2_locations_rearr = c2_locations_rearr_batch[i]

            idx = torch.argmax(c2_locations_rearr).item()

            # Top indices of C2
            c2_row_min = idx // crop_size
            c2_col_min = idx % crop_size

            # Top indices of C1 decomposition
            c1_decomp_row_min = torch.argmax(c1_decomp_gt).item() // (decomposition_factor * crop_size)
            c1_decomp_col_min = torch.argmax(c1_decomp_gt).item() % (decomposition_factor * crop_size)

            # Number of steps
            num_row = torch.sum(torch.any(c2_locations_rearr.reshape(crop_size, crop_size), dim=1)).item()
            num_col = torch.sum(c2_locations_rearr.reshape(crop_size, crop_size)[c2_row_min]).item()

            for r in range(num_row):
                for c in range(num_col):
                    c2_idx = (c2_row_min + r) * crop_size + (c2_col_min + c)
                    effective_idx = r * crop_size + c
                    effective_r = effective_idx // (crop_size * 2)
                    effective_c = effective_idx % (crop_size * 2)

                    c1_decomp_idx = int((c1_decomp_row_min + effective_r) * (crop_size * 2) + (c1_decomp_col_min + effective_c))
                    decomp_barlow_labels_batch[i, c2_idx, c1_decomp_idx] = True

        return decomp_barlow_labels_batch

    comp_barlow_labels = process_comp_barlow_labels_batch(c2_comp_gt, c1_locations_batch, crop_size=crop_size)
    decomp_barlow_labels = process_decomp_barlow_labels_batch(c1_decomp_gt, c2_locations_batch, crop_size=crop_size)
    return comp_barlow_labels, decomp_barlow_labels

class LocalCompDecompCrop() :
    def __init__(self, size, patch_size, crop_size):
        self.size = size
        self.patch_size = patch_size
        self.crop_size = crop_size
    def __call__(self, img):
        # Coordinates of crops with granularity of patch_size
        c1_coords, c2_coords = get_coordinates(
            img, patch_size=self.patch_size, crop_size=self.crop_size
        )
        # Actual crops of image
        c1, c2 = crop(img, c1_coords, c2_coords, crop_size=self.crop_size, patch_size=self.patch_size)
        # Resize
        new_size = self.patch_size * self.crop_size
        # c1 = np.transpose(c1, (1, 2, 0))
        # c2 = np.transpose(c2, (1, 2, 0))
        c1 = cv2.resize(c1, (new_size, new_size))
        c2 = cv2.resize(c2, (new_size, new_size))
        # c1 = np.transpose(c1, (2, 0, 1))
        # c2 = np.transpose(c2, (2, 0, 1))
        # # Patches
        # c1_patches = get_patches(c1, self.patch_size, self.crop_size)
        # c2_patches = get_patches(c2, self.patch_size, self.crop_size)

        concat_input = np.concatenate((c1, c2), axis=2)
        return concat_input, c1_coords, c2_coords, (2, 2)


if __name__ == "__main__":
    pass
    # patch_size = 32
    # crop_size = 14
    # img = np.random.rand(3, 1024, 1024).astype(np.float32)
    # for _ in range(1):
    #     c1_coords, c2_coords = get_coordinates(
    #         img, patch_size=patch_size, crop_size=crop_size
    #     )
    #     print(c1_coords, c2_coords, "\n")
    #     c1, c2 = crop(img, c1_coords, c2_coords, crop_size=crop_size)
    #     # Resize
    #     new_size = patch_size * crop_size
    #     c1 = np.transpose(c1, (1, 2, 0))
    #     c2 = np.transpose(c2, (1, 2, 0))
    #     c1 = cv2.resize(c1, (new_size, new_size))
    #     c2 = cv2.resize(c2, (new_size, new_size))
    #     c1 = np.transpose(c1, (2, 0, 1))
    #     c2 = np.transpose(c2, (2, 0, 1))
    #     # Patches
    #     c1_patches = get_patches(c1, patch_size, crop_size)
    #     c2_patches = get_patches(c2, patch_size, crop_size)
    #     c1_patches = torch.tensor(c1_patches, dtype=torch.float32)
    #     c2_patches = torch.tensor(c2_patches, dtype=torch.float32)
    #     # print(c1_patches.shape, c2_patches.shape)
    #     # Embeddings
    #     c1_embs = get_embeddings(c1_patches, SimpleModel())
    #     c2_embs = get_embeddings(c2_patches, SimpleModel())
    #     # print(c1_embs.shape, c2_embs.shape)
    #     c1_rearr = rearrange_embeddings(c1_embs, crop_size=crop_size)
    #     c1_comp = composition(c1_rearr, model=CompositionHead())
        # print(c1_comp.shape)