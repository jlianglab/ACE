# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys

import torch
import torch.distributed as dist
# from torch._six import inf
from torch import inf
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    # for i, (name, par) in enumerate(model.named_parameters()):
    #         print(i, name)
    
    # NIHchest pretrained
    # state_dict = checkpoint['model']


        # sys.exit(1)

    # # pec
    if config.PRETRAIN_MODE == 'only_pec':
        state_dict = checkpoint['student']
        model_keys = list(state_dict.keys())
        for i in range(-1,-10,-1): # range(start, stop, step)
            state_dict.pop(model_keys[i])
        model_keys = list(state_dict.keys())

        for i in range(len(model_keys)):
            key = model_keys[i]
            newkey = key[18:]
            state_dict[newkey] = state_dict.pop(key)
        with open('/home/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/pec_keys.txt', 'w') as f:
            for i in range(len(list(state_dict.keys()))):
                f.writelines(list(state_dict.keys())[i]+'\n')
        # sys.exit(1)

    # pec_popar
    elif config.PRETRAIN_MODE in ['popar_pec', 'simmim_global']:
        state_dict = checkpoint['student']
        model_keys = list(state_dict.keys())
        for i in range(-1,-14,-1): # range(start, stop, step)
            print(model_keys[i])
            state_dict.pop(model_keys[i])
        model_keys = list(state_dict.keys())

        # state_dict = {k.replace("module.swin_model", "backbone"): v for k, v in state_dict.items()}

        for i in range(len(model_keys)):
            key = model_keys[i]
            newkey = key[18:]
            state_dict[newkey] = state_dict.pop(key)
        with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/simmim_global.txt', 'w') as f:
            for i in range(len(list(state_dict.keys()))):
                f.writelines(list(state_dict.keys())[i]+'\n')
        # sys.exit(1)
    elif config.PRETRAIN_MODE in ['popar_adar','popar_adoc','popar_adocar','popar_adodocar','popar_odar','popar_odoc','popar_odocar','simmim','popar','popar_cyclic','popar^','popar^_cyclic']:
        state_dict = checkpoint['student']
        model_keys = list(state_dict.keys())
        for i in range(-1,-5,-1): # range(start, stop, step)
            state_dict.pop(model_keys[i])
        model_keys = list(state_dict.keys())

        for i in range(len(model_keys)):
            key = model_keys[i]
            newkey = key[18:]
            state_dict[newkey] = state_dict.pop(key)
    #     with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/simmim_ddp_keys.txt', 'w') as f:
    #         for i in range(len(list(state_dict.keys()))):
    #             f.writelines(list(state_dict.keys())[i]+'\n')
    # sys.exit(1)
    elif config.PRETRAIN_MODE in ['local_infonce']:
        state_dict = checkpoint['student']
        model_keys = list(state_dict.keys())
        # for i in range(-1,-14,-1): # range(start, stop, step)
        #     print(model_keys[i])
        #     state_dict.pop(model_keys[i])
        # model_keys = list(state_dict.keys())

        # state_dict = {k.replace("module.swin_model", "backbone"): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif config.PRETRAIN_MODE in ['seg_simmim_global', 'seg_simmim', 'simmim_global_infonce', 'simmim_global_barlow','popar_pec_seg', 'seg_popar']:
        try:
            state_dict = checkpoint['student']
        except:
            state_dict = checkpoint['model']
        state_dict = {k.replace("module.swin_model", "backbone"): v for k, v in state_dict.items()}

    elif config.PRETRAIN_MODE in ['simmim_global_infonce', 'simmim_global_barlow', 'l1loss_local', 'cosineloss_local']:
        state_dict = checkpoint['student']
        model_keys = list(state_dict.keys())

        state_dict = {k.replace("module.swin_model.", ""): v for k, v in state_dict.items()}
        with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/simmim_global.txt', 'w') as f:
            for i in range(len(list(state_dict.keys()))):
                f.writelines(list(state_dict.keys())[i]+'\n')
    elif config.PRETRAIN_MODE in ['simmim_global_barlow']:
        state_dict = checkpoint['teacher']
        model_keys = list(state_dict.keys())

        state_dict = {k.replace("module.swin_model.", ""): v for k, v in state_dict.items()}

    elif config.PRETRAIN_MODE in ['simmim_imagenet1k']: 
        state_dict = checkpoint['model']
        state_dict = {k.replace("encoder", "backbone"): v for k, v in state_dict.items()}
        checkpoint = remap_pretrained_keys_swin(model, state_dict, logger)
    
    elif config.PRETRAIN_MODE in ['swin_imagenet1k']:
        state_dict = checkpoint['model']
        # with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/swin_imagenet1k.txt', 'w') as f:
        #     for i in range(len(list(state_dict.keys()))):
        #         f.writelines(list(state_dict.keys())[i]+'\n')
        state_dict = {"backbone."+k: v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

    elif config.PRETRAIN_MODE in ['swin_nih']:
        state_dict = checkpoint['model']
        state_dict = {k.replace("module.", "backbone."): v for k, v in state_dict.items()}
        

    elif config.PRETRAIN_MODE in ['swinv2_imagenet1k']:
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'relative_coords_table' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'relative_position_index' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        # state_dict = {"module.backbone."+k: v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['swinv2_nih']:
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'relative_coords_table' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'relative_position_index' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['seg_swinv2_imagenet1k']:
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'relative_coords_table' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'relative_position_index' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        state_dict = {"backbone."+k: v for k, v in state_dict.items()}

    elif config.PRETRAIN_MODE in ['seg_swinv2_nih']:
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if 'relative_coords_table' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'relative_position_index' not in k}
        state_dict = {k: v for k, v in state_dict.items() if 'attn_mask' not in k}
        state_dict = {k.replace("module.", "backbone."): v for k, v in state_dict.items()}

        # ssl genesis
    elif config.PRETRAIN_MODE in ['ssl_genesis_50epc', 'ssl_genesis_100epc', 'ssl_genesis_150epc','fromscratch_extrap_shuffle_compdecomp_consis_100epc','fromMIM_extrap_shuffle_compdecomp_25epc',\
                                  'onebranch_3component_8epc','onebranch_3component_24epc','large_fromMIM_extrap_shuffle_consis_compdecomp','large_fromscratch_extrap_shuffle_compdecomp_consis']:
        try:
            state_dict = checkpoint['teacher']
        except:
            state_dict = checkpoint['model']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    
    elif config.PRETRAIN_MODE in ['unique_multiscale_consis_compdecomp_100epc', 'unique_multiscale_consis_compdecomp_150epc', 'fromIN_unique_multiscale_consis_compdecomp_50epc','fromIN_unique_multiscale_consis_compdecomp_25epc',\
                                    'fromIN_unique_multiscale_consis_compdecomp_100epc','swinv2_fromIN_unique_multiscale_consis_compdecomp_50epc', 'swinv2_fromIN_unique_multiscale_consis_compdecomp_25epc', 'large_swinv2_fromIN_unique_multiscale_consis_compdecomp_20epc']:
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['fromIN_uniqueness_multigranu_big_local_crop_150epc']:
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
    
    elif config.PRETRAIN_MODE in ['seg_fromIN_unique_multiscale_consis_compdecomp_50epc','seg_large_swinv2_fromIN_unique_multiscale_consis_compdecomp_20epc','seg_swinv2_fromIN_unique_multiscale_consis_compdecomp_50epc',\
                                'seg_fromIN_unique_multiscale_consis_compdecomp_100epc', 'seg_unique_multiscale_consis_compdecomp_100epc', 'seg_fromIN_uniqueness_multigranu_big_local_crop_150epc',\
                                'seg_consistency_fromIN', 'seg_fromIN_compdecomp_50epc']:
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    
    
        
    elif config.PRETRAIN_MODE in ['uniqueness_fronDINOv2_vitb', 'uniqueness_fronDINOv2_vitb_100epc', 'uniqueness_fronDINOv2_vitb_150epc']:
        try:
            state_dict = checkpoint['teacher']
        except:
            state_dict = checkpoint['model']
        state_dict = {k.replace("backbone.", "base_model."): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['vitb_fromDINOv2_unique_multiscale_consis_compdecomp_50epc','vitb_fromDINOv2_unique_multiscale_consis_compdecomp_100epc','vitb_fromDINOv2_3unique_multiscale_consis_compdecomp_50epc']: 
        try:
            state_dict = checkpoint['teacher']
        except:
            state_dict = checkpoint['model']
        state_dict = {k.replace("module.backbone.", "base_model."): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['seg_vitb_fromDINOv2_unique_multiscale_consis_compdecomp_50epc', 'seg_uniqueness_fronDINOv2_vitb_100epc', 'seg_vitb_fromDINOv2_3unique_multiscale_consis_compdecomp_50epc']:
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("module.backbone.", "backbone."): v for k, v in state_dict.items()}

    # elif config.PRETRAIN_MODE in ['seg_ssl_genesis_150epc']: # simmim pre-trained patch size 6 mismatch with fine-tuned patch size 7
    #     state_dict = checkpoint['teacher']
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif config.PRETRAIN_MODE in ['seg_fromMIM_extrap_shuffle_compdecomp_25epc','seg_onebranch_3component_24epc','seg_large_fromMIM_extrap_shuffle_consis_compdecomp','seg_fromscratch_extrap_popar_compdecomp_consis',\
                                  ]: # simmim pre-trained patch size 6 mismatch with fine-tuned patch size 7
        state_dict = checkpoint['teacher']
    
    # elif config.PRETRAIN_MODE in ['seg_ark6']:
    #     state_dict = checkpoint

    elif config.PRETRAIN_MODE in ['compose_12N', 'compose_12N_infonce', 'simmim_compose12N', 'simmim_compose12N_200ep', 'byol', 'comp_decomp', 'clip_global_simmim','clip_global','clip','contrast_12n_global',\
                                  'simmim_compose12N_infonce','dino','contrast_12n_global_inequal','contrast', 'ablation_GloLocConsis_100epc', 'random_mask_100epc', 'ACE_fromIN_swinv2', 'ACE_fromIN_largescale',\
                                    'ACE_fromIN_largescale_50epc','symmetry_global_100epc','symmetry_local_50epc','ACE_fromscratch_100epc_swinv2','consistency_fromIN']: # simmim pre-trained patch size 6 mismatch with fine-tuned patch size 7
        state_dict = checkpoint['student']
        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}

    elif config.PRETRAIN_MODE in ['seg_compose_12N', 'seg_compose_12N_infonce', 'seg_byol', 'seg_dino','vit_seg_selfpatch','seg_contrast_12n_global','seg_contrast_12n_global_inequal','seg_contrast_12n_inequal','seg_contrast',\
                                  'seg_extrap_100epc','seg_ablation_GloLocConsis_100epc','seg_hierar_comp_decomp_100epc','seg_patch_shuffle_student_teacher_100epc','seg_ACE_fromIN_swinv2','seg_ACE_fromIN_largescale_50epc']:
        state_dict = checkpoint['student']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['seg_vitb_nih']:
        state_dict = checkpoint['model']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    elif config.PRETRAIN_MODE in ['vit_seg_droppos']:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace("module.", "backbone."): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]


    elif config.PRETRAIN_MODE in ['seg_ce_clip_itm_local_vitbps16','seg_ce_clip_itm_vitbps16','seg_ce_clip_vitbps16','seg_kad_vit','seg_devide']:
        state_dict = checkpoint['image_encoder']
        state_dict = {k.replace("vit", "backbone"): v for k, v in state_dict.items()}

        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
    elif config.PRETRAIN_MODE in ['droppos_vit']:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]
    elif config.PRETRAIN_MODE in ['vit_seg_dino','contrast_12n_global_inequal_vit_seg', 'seg_vit_extrap_shuffle_compdecomp']:
        state_dict = checkpoint['student']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]
    elif config.PRETRAIN_MODE in ['contrast_12n_global_inequal_vit','dino_vit','selfpatch_vit']:
        state_dict = checkpoint['student']
        state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]

    elif config.PRETRAIN_MODE in ['vit_extrap_shuffle_compdecomp_50epc','vit_extrap_shuffle_compdecomp_100epc','vit_extrap_shuffle_compdecomp_150epc']:
        state_dict = checkpoint['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]
    elif config.PRETRAIN_MODE in ['seg_vit_extrap_shuffle_compdecomp_100epc']:
        state_dict = checkpoint['teacher']
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]

    elif config.PRETRAIN_MODE in ['ijepa_vit', 'ijepa_ImageNet-1K']:
        state_dict = checkpoint['encoder']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        pos_embed = [k for k in state_dict.keys() if "pos_embed" in k]
        for k in pos_embed:
            del state_dict[k]

    elif config.PRETRAIN_MODE in ['adam_resnet_seg', 'seg_ark6']:
        state_dict = checkpoint
        state_dict = {"backbone."+k: v for k, v in state_dict.items()}
    elif config.PRETRAIN_MODE in ['adam_resnet']:
        state_dict = checkpoint
    elif config.PRETRAIN_MODE in ['adam-v2_seg']:
        state_dict = checkpoint['teacher']


    elif config.PRETRAIN_MODE in ['ark_lamps_scratch_alternative_global_distillation_wo_teacher_mlp_sumMSE','ark_lamps_scratch_alternative_local_distillation_wo_teacher_mlp_sumMSE']: # simmim pre-trained patch size 6 mismatch with fine-tuned patch size 7

        state_dict = checkpoint['model']
        state_dict = {k.replace("module.student_model.", ""): v for k, v in state_dict.items()}

        # sys.exit(1)


    # # delete relative_position_index since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]

    # # delete relative_coords_table since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]

    # delete attn_mask since we always re-init it
    # attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    # for k in attn_mask_keys:
    #     del state_dict[k]

    # # bicubic interpolate relative_position_bias_table if not match
    # relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    # for k in relative_position_bias_table_keys:
    #     relative_position_bias_table_pretrained = state_dict[k]
    #     relative_position_bias_table_current = model.state_dict()[k]
    #     L1, nH1 = relative_position_bias_table_pretrained.size()
    #     L2, nH2 = relative_position_bias_table_current.size()
    #     if nH1 != nH2:
    #         logger.warning(f"Error in loading {k}, passing......")
    #     else:
    #         if L1 != L2:
    #             # bicubic interpolate relative_position_bias_table if not match
    #             S1 = int(L1 ** 0.5)
    #             S2 = int(L2 ** 0.5)
    #             relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
    #                 relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
    #                 mode='bicubic')
    #             state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # # bicubic interpolate absolute_pos_embed if not match
    # absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    # for k in absolute_pos_embed_keys:
    #     # dpe
    #     absolute_pos_embed_pretrained = state_dict[k]
    #     absolute_pos_embed_current = model.state_dict()[k]
    #     _, L1, C1 = absolute_pos_embed_pretrained.size()
    #     _, L2, C2 = absolute_pos_embed_current.size()
    #     if C1 != C1:
    #         logger.warning(f"Error in loading {k}, passing......")
    #     else:
    #         if L1 != L2:
    #             S1 = int(L1 ** 0.5)
    #             S2 = int(L2 ** 0.5)
    #             absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
    #             absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
    #             absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
    #                 absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
    #             absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
    #             absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
    #             state_dict[k] = absolute_pos_embed_pretrained_resized

    # # check classifier, if not match, then re-init classifier to zero
    # head_bias_pretrained = state_dict['head.bias']
    # Nc1 = head_bias_pretrained.shape[0]
    # Nc2 = model.head.bias.shape[0]
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         del state_dict['head.weight']
    #         del state_dict['head.bias']
    #         logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    # # load popar pretrained model时删除
    # del state_dict['head.weight']
    # del state_dict['head.bias']
    if 'head.weight' in state_dict:
        del state_dict['head.weight']
    if 'head.bias' in state_dict:
        del state_dict['head.bias']

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()
    # sys.exit(1)


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                #   'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_auc': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")



def save_checkpoint_last(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_auc': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'last.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")



def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class DiceLoss(nn.Module):
    """
    Implements the dice loss function.
    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
    def __init__(self, ignore_index = 3): # 无用像素是其他类别，mask像素值设为3
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5 # 防止分母为0加上此参数

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = torch.unsqueeze(labels, 1) # labels维度batchsize*1*h*w
        num_classes = logits.shape[1] # 分割类别
        # mask = (labels != self.ignore_index)
        # mask = labels
        # logits = logits * mask
        single_label_list = []

        for c in range(num_classes):
            single_label = (labels == c)
            single_label = torch.squeeze(single_label, 1)
            single_label_list.append(single_label)
        labels_one_hot = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
        logits = F.softmax(logits, dim = 1) # logits维度batchsize*4*h*w
        dims = (0,2,3) # 压缩0，2，3这三个维度，最后得到的loss是一个长度为4的一维向量，其值分别为4个类别的dice
        intersection = torch.sum(logits * labels_one_hot, dims)
        cardinality = torch.sum(logits + labels_one_hot, dims)
        dice_score = (2. * intersection / (cardinality + self.eps))
        dice_loss = (1-dice_score).mean()
        return dice_loss, dice_score # 返回四个类别的dice_score
    
    


def save_model_wo_conf(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    }
    torch.save(state, save_file)
    del state