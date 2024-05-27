import torch
import numpy as np
import random
import cv2
import torch.nn as nn
import torch.nn.functional as F

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

class CompositionHead(nn.Module):
    def __init__(self, composition_factor=4, embedding_size=3):
        super(CompositionHead, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(composition_factor * embedding_size, embedding_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecompositionHead(nn.Module):
    def __init__(self, composition_factor=4, embedding_size=3):
        super(DecompositionHead, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_size, composition_factor * embedding_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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


def get_embeddings(patches, model):
    embeddings = []
    for patch in patches:
        embeddings.append(model(patch).detach().numpy())
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings

def prep_composition(embeddings, composition_factor=4):
    num_groups = embeddings.shape[0] // composition_factor
    print(embeddings,"\n")
    groups = embeddings.reshape(num_groups, embeddings.shape[1] * composition_factor)
    print(groups)
    groups = torch.tensor(groups, dtype=torch.float32)
    # composed = model(groups)
    return groups


def rearrange_embeddings(input, crop_size=14):
    input_shape = input.size()
    last_two_dims = input_shape[-2:]

    input_tensor_reshaped = input.view(-1, *last_two_dims)

    input_slices = torch.split(input_tensor_reshaped, 1, dim=0)

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
        # print(final_index)
        # Rearrange embeddings using fancy indexing
        rearranged_embeddings = embeddings[final_index]
        # ipdb.set_trace()
        return rearranged_embeddings

    results = torch.cat([apply_logic(embeddings.squeeze(0)) for embeddings in input_slices], dim=0)
    reshaped = results.view(*input_shape[:-2], *last_two_dims)
    return reshaped

# Function to get Composition labels
def get_comp_gt(c2_overlap_gt : torch.Tensor, crop_size = 14):
    # Input : c2_overlap_gt : (196, 1)
    # Output : comp_embeddings : (49, 1)
    ''' IF we view [196,1] as [14,14]:
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 1 1 1 1 1
    0 0 0 0 0 0 0 1 1 1 1 1 1
    0 0 0 0 0 0 0 1 1 1 1 1 1
    0 0 0 0 0 0 0 1 1 1 1 1 1
    0 0 0 0 0 0 0 1 1 1 1 1 1
    0 0 0 0 0 0 0 1 1 1 1 1 1
    The [7,7] view of the output will look like this:
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 1 1 1
    0 0 0 0 1 1 1
    0 0 0 0 1 1 1
    '''
    comp_embeddings = torch.empty((0,))
    for i in range(0, c2_overlap_gt.shape[0], crop_size * 2):
        for j in range(0, crop_size, 2):
            if c2_overlap_gt[i + j]:
                comp_embeddings = torch.cat((comp_embeddings, torch.tensor([1])), dim=0)
            else:
                comp_embeddings = torch.cat((comp_embeddings, torch.tensor([0])), dim=0)
    return comp_embeddings

# Function to get Decomposition labels
def get_decomp_gt(c1_overlap_gt : torch.Tensor):
    # Output : c1_overlap_gt : (196, 1)
    # Input : decomp_embeddings : (784, 1)
    '''
    Coverts an input of [0,1,0,1] to [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
    '''
    c1_decomp_embeddings = torch.zeros((c1_overlap_gt.shape[0] * 4, 1))
    # ipdb.set_trace()
    for i, item in enumerate(c1_overlap_gt):
        c1_decomp_embeddings[i * 4 : i * 4 + 4] = item
    return c1_decomp_embeddings

def get_comp_decomp_barlow_labels(c1_decomp_gt : torch.Tensor,
                                    c2_comp_gt : torch.Tensor,
                                    c1_locations : torch.Tensor,
                                    c2_locations : torch.Tensor,
                                    crop_size=14):
    # Input : c1_decomp_gt : (784, 1) , c2_comp_gt : (49, 1), c1_locations : (196, 1), c2_locations : (196, 1)\
    # Output : comp_barlow_labels : (196, 49) , decomp_barlow_labels : (196, 784)

    def process_comp_barlow_labels(c2_comp_gt : torch.Tensor,
                                    c1_locations : torch.Tensor,
                                    crop_size = crop_size):
        # Input : c2_comp_gt : (49, 1)
        # Input : c2_locations : (196, 1)
        # Output : comp_barlow_labels : (196, 49)
        '''
        We will see which of the 196 embeddings in C1 ground truths correspond to the positive embeddings in composition embeddings
        '''
        # print("In composition: ")
        comp_barlow_labels = torch.zeros((c1_locations.shape[0], c2_comp_gt.shape[0]))
        idx = torch.argmax(c1_locations).item()
        # Top indices of C1
        c1_row_min = idx // crop_size
        c1_col_min = idx % crop_size
        # Top indices of C2 composition
        c2_comp_row_min = torch.argmax(c2_comp_gt).item() // (crop_size / 2)
        c2_comp_col_min = torch.argmax(c2_comp_gt).item() % (crop_size / 2)
        # print(c2_comp_row_min, c2_comp_col_min)
        # Number of steps
        # print("c1_row_min: ", c1_row_min, "c1_col_min: ", c1_col_min)
        # print("c2_comp_row_min: ", c2_comp_row_min, "c2_comp_col_min: ", c2_comp_col_min)
        num_row = torch.sum(torch.any(c1_locations.reshape(14,14), dim=1)).item()
        num_col = torch.sum(c1_locations.reshape(14,14)[c1_row_min]).item()
        # print("num_row: ", num_row, "num_col: ", num_col)
       
        for r in range(num_row):
            for c in range(num_col):
                # print(r,c)
                c1_idx = (c1_row_min + r) * crop_size + (c1_col_min + c)
                c2_comp_idx = int((c2_comp_row_min + r) * (crop_size // 2) + (c2_comp_col_min + c))
                comp_barlow_labels[c1_idx][c2_comp_idx] = True
        return comp_barlow_labels

    def process_decomp_barlow_labels(c1_decomp_gt, c2_locations, decomposition_factor=4, crop_size=crop_size):
        # Input : c1_decomp_gt : (784, 1)
        # Input : c2_locations : (196, 1)
        # Output : decomp_barlow_labels : (196, 784)
        '''
        We will see which of the 196 embeddings in C2 ground truths correspond to the positive embeddings in decomposition embeddings
        '''
        # print("In decomposition: ")
        decomp_barlow_labels = torch.zeros((c2_locations.shape[0], c1_decomp_gt.shape[0]))
        c2_locations_rearr = rearrange_embeddings(c2_locations)
        idx = torch.argmax(c2_locations_rearr).item()
        # Top indices of C2
        c2_row_min = idx // crop_size
        c2_col_min = idx % crop_size
        # Top indices of C1 decomposition
        c1_decomp_row_min = torch.argmax(c1_decomp_gt).item() // (decomposition_factor * crop_size)
        c1_decomp_col_min = torch.argmax(c1_decomp_gt).item() % (decomposition_factor * crop_size)
        # Number of steps
        # print("c2_row_min: ", c2_row_min, "c2_col_min: ", c2_col_min)
        # print("c1_decomp_row_min: ", c1_decomp_row_min, "c1_decomp_col_min: ", c1_decomp_col_min)
        num_row = torch.sum(torch.any(c2_locations_rearr.reshape(14,14), dim=1)).item()
        num_col = torch.sum(c2_locations_rearr.reshape(14,14)[c2_row_min]).item()
        # print("num_row: ", num_row, "num_col: ", num_col)
        # print("")
        for r in range(num_row):
            for c in range(num_col):
                # print(r, c)
                c2_idx = (c2_row_min + r) * crop_size + (c2_col_min + c)
                effective_idx = r * crop_size + c
                effective_r = effective_idx // (crop_size * 2)
                effective_c = effective_idx % (crop_size * 2)
                # print(effective_r, effective_c)
                c1_decomp_idx = int((c1_decomp_row_min + effective_r) * (crop_size * 2)  + (c1_decomp_col_min + effective_c))
                decomp_barlow_labels[c2_idx][c1_decomp_idx] = True
        return decomp_barlow_labels

    comp_barlow_labels = process_comp_barlow_labels(c2_comp_gt, c1_locations, crop_size=crop_size)
    decomp_barlow_labels = process_decomp_barlow_labels(c1_decomp_gt, c2_locations)
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