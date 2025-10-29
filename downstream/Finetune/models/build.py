# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .simmim import build_simmim
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
import torchvision.models as models
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoModel


def build_model(config, is_pretrain=False):
    backbone = config.BACKBONE
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if backbone == 'swin_base':
        if model_type == 'swin':
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    norm_layer=layernorm,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    fused_window_process=config.FUSED_WINDOW_PROCESS)
        elif model_type == 'swinv2':
            model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                    in_chans=config.MODEL.SWINV2.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                    depths=config.MODEL.SWINV2.DEPTHS,
                                    num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                    window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWINV2.APE,
                                    patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
        elif model_type == 'swin_moe':
            model = SwinTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                                    depths=config.MODEL.SWIN_MOE.DEPTHS,
                                    num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                                    window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN_MOE.APE,
                                    patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                                    mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                                    init_std=config.MODEL.SWIN_MOE.INIT_STD,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                    moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                                    num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                    top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                                    capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                                    cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                                    normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                                    use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                                    is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                                    gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                                    cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                                    cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                    moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                                    aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)
        elif model_type == 'swin_mlp':
            model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                            depths=config.MODEL.SWIN_MLP.DEPTHS,
                            num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                            window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                            mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            ape=config.MODEL.SWIN_MLP.APE,
                            patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")
    elif backbone=='vit_base':
        model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()

    elif backbone=="vit_base_patchsize16":
        if config.PRETRAIN_MODE in ['ce_clip_itm_vitbps16','ce_clip_vitbps16']:
            model = ViTForImageClassification.from_pretrained(
                '/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/vit-base-patch16-224-in21k',
                num_labels=config.MODEL.NUM_CLASSES)
        elif config.PRETRAIN_MODE in ['RAD-DINO']:
            base_model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino')
            model = ClassificationModel(base_model, num_labels=config.MODEL.NUM_CLASSES)
        elif 'DINOv2' in config.PRETRAIN_MODE:
            base_model = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/dinov2-base')
            model = ClassificationModel(base_model, num_labels=config.MODEL.NUM_CLASSES)
        else:
            model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()

    elif backbone=="vit_huge_patchsize14":
        model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                                  patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    elif backbone=='resnet50':
        model = models.__dict__['resnet50'](pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, config.MODEL.NUM_CLASSES)

    return model




class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, x):
        outputs = self.base_model(x)
        pooled_output = outputs.pooler_output  # 获取池化后的输出
        logits = self.classifier(pooled_output)
        return logits
