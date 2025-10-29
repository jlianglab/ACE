import argparse
import torch
import torchvision.transforms
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from ViT import vit_new, vit_base
from swin_transformer import SwinTransformer
from transformers import ViTImageProcessor, ViTModel, AutoModel, AutoConfig
import ipdb

class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            print(model_type)
            self.model = ViTExtractor.create_model(model_type)

        # self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        # self.p = self.model.patch_embed.patch_size
        # self.stride = self.model.patch_embed.proj.stride
        self.p = self.model.embeddings.patch_embeddings.patch_size[0] # 14
        self.stride = self.model.embeddings.patch_embeddings.projection.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        #if 'dino' in model_type:
            # = torch.hub.load('facebookresearch/dino:main', model_type)
        # else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
        #     temp_model = timm.create_model(model_type, pretrained=True)
        # model_type_dict = {
        #     'vit_small_patch16_224': 'dino_vits16',
        #     'vit_small_patch8_224': 'dino_vits8',
        #     'vit_base_patch16_224': 'dino_vitb16',
        #     'vit_base_patch8_224': 'dino_vitb8'
        # }
        from timm.models.vision_transformer import VisionTransformer, _cfg
        from functools import partial
        
        if model_type == 'dinov2':
            config_model = AutoConfig.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino')
            model = AutoModel.from_config(config_model)
            checkpoint = torch.load('/mnt/nvme1n1/zhouziyu/ACE_journal/uniqueness/pretrained_weight/vitb_fromDINOv2_518/checkpoint0050.pth', map_location='cpu')
        else:
            model = vit_new()
            # model = vit_base()
            # model = SwinTransformer(img_size=448, embed_dim=128, depths=[ 2, 2, 18, 2 ], num_heads=[ 4, 8, 16, 32 ])
            # model = VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12,
                                # mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                # drop_rate=0,drop_path_rate=0.1, in_chans = 3)
            # model = timm.create_model('vit_base_patch16_224', pretrained=False)
            #temp_state_dict = temp_model.state_dict()
            # checkpoint = torch.load("/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/ACEv4/pretrained_weight/clstoken_global_vit_ps16/checkpoint.pth", map_location='cpu')
            # checkpoint = torch.load("/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/compose/matrixcompdecompmlp_overlapglobal_vit_checkpoint0150.pth", map_location='cpu')
            checkpoint = torch.load("/mnt/nvme1n1/zhouziyu/sslgenesis_vit/pretrained_weight/vit_extrap_shuffle_compdecomp/checkpoint0052.pth", map_location='cpu')
            # checkpoint = torch.load("/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dino/dino_vit_checkpoint0300.pth", map_location='cpu')      # checkpoint = torch.load('/sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/dropPos_vit-b32_448/droppos.pth', map_location='cpu')
     
        try:
            checkpoint = checkpoint['student']
        except:
            checkpoint = checkpoint['state_dict']
        # checkpoint_model = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint_model = {k.replace("vit_model.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
       # del checkpoint_model['head.weight']
        # del checkpoint_model['head.bias']
        checkpoint_model = {k.replace("swin_model.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("module.backbone.", ""): v for k, v in checkpoint_model.items()}
        with open('vit.txt', 'w') as f:
            for i in range(len(list(checkpoint_model.keys()))):
                f.writelines(list(checkpoint_model.keys())[i]+'\n')
        
        # for key in model.state_dict().keys():
        #     print(key)
        # for key in checkpoint_model.keys():
        #     print(key)
        #     if key in model.state_dict().keys():
        #         print(key)
        #         model.state_dict()[key].copy_(checkpoint_model[key])
        #         print("Copying {} <---- {}".format(key, key))
        #     else:
        #         #pass
        #         print("Key {} is not found".format(key))
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


        #model.load_state_dict(checkpoint_model)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        # patch_size = 32#model.patch_embed.patch_size
        patch_size = 16#model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        # print(model.patch_embed.proj.stride) # vitb_patchsize32_224
        # model.patch_embed.proj.stride = stride
        
        print(model.embeddings.patch_embeddings.projection.stride) # dinov2
        model.embeddings.patch_embeddings.projection.stride = stride

        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")
        # ipdb.set_trace()

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            # qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            # self._feats.append(qkv[facet_idx]) #Bxhxtxd
            
            # facet == 'key' 
            qkv = module.key(input).reshape(B, N, module.num_attention_heads, C // module.num_attention_heads).permute(0, 2, 1, 3) # dinov2
            # ipdb.set_trace()
            self._feats.append(qkv)
            # self._feats.append(qkv if isinstance(qkv, torch.Tensor) else qkv[0])
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        # vit
        # for block_idx, block in enumerate(self.model.blocks):
        for block_idx, block in enumerate(self.model.encoder.layer):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    # self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                    self.hook_handlers.append(block.attention.attention.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    # self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                    self.hook_handlers.append(block.attention.attention.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")
                
        # # swin
        # for layer_idx, layer in enumerate(self.model.layers):
        #     self.current_layer_idx = layer_idx
        #     self.current_block_idx = -1
        #     for block_idx, block in enumerate(layer.blocks):
        #         self.current_block_idx = block_idx
        #         if layer_idx == 3 and block_idx == 1:  # Layer 3, Block 1
        #             if facet == 'token':
        #                 self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
        #             elif facet == 'attn':
        #                 self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
        #             elif facet in ['key', 'query', 'value']:
        #                 self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
        #             else:
        #                 raise TypeError(f"{facet} is not a supported facet.")
                

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        # print(layers)
        # ipdb.set_trace()
        self._register_hooks(layers, facet)
        print(batch.shape)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)

        # ipdb.set_trace()
        # print(self.p) # 32
        # print(self.stride) # (8,8)
        # self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1]) # vit
        self.num_patches = (H// self.p, W // self.p)  # dinov2
        # self.num_patches = (1 + (H - 32) // 8, 1 + (W - 32) // 8) # swin
        # print(self.num_patches)
        # print(self._feats.shape)

        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy
        # print('x',x.shape)
        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        # print('bin_x',bin_x.shape)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # print('reshaped_bin_x', bin_x.shape)
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        # print(self._feats.shape)
        print('feats',self._feats[0].shape)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:

            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)

        else:
            desc = self._log_bin(x)
            # print('desc',desc.shape)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        #assert self.model_type == "dino_vits16", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        # ipdb.set_trace()
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor extraction.')
    parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
    parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    parser.add_argument('--load_size', default=518, type=int, help='load size of the input image.') # dinov2:518
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                              small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dinov2 | dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")

    args = parser.parse_args()

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = ViTExtractor(args.model_type, args.stride, device=device)
        image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size)
        print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")
        descriptors = extractor.extract_descriptors(image_batch.to(device), args.layer, args.facet, args.bin)
        print(f"Descriptors are of size: {descriptors.shape}")
        torch.save(descriptors, args.output_path)
        print(f"Descriptors saved to: {args.output_path}")
