

# CUDA_VISIBLE_DEVICES="3" python -m torch.distributed.launch --nproc_per_node=1 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 8
# CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 16

# CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28300 main.py --arch vit_base --batch_size_per_gpu 8

# CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28300 main.py --arch swinv2_base --batch_size_per_gpu 8

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from infonce import *
import utils
import vision_transformer as vits
import models.swin_transformer as swins
from vision_transformer import DINOHead,SimMIM_head,SimMIM_head_SWIN, DenseHead
from ImageFolder_vindr import ImageFolder_vindr,ChestX_ray14,ShenzhenCXR,LDPolyp
from config import config
from config import config_swinv2
from config import update_config
from config import save_config
from models import build_model
from transforms import MultiCropTrainDataTransform, DataAugmentationDINO
from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
import pandas as pd
from transformers import AutoModel
import ipdb
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='swin_base', type=str,
        choices=['swin_base', 'vit_base','swinv2_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--image_size', default=448, type=int, help="""resolution of global crops.""") # 448 for swinv1, 512 for swinv2, 518 for vitb

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0.8, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=20, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.8, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=4, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--data_granularity', default=(1.,0.75,0.5,0.25,0.125,0.375,0.625,0.875,0.9375,0.6875,0.4375,0.1875,0.2188,0.4688,0.7188,0.9688), 
                        type=float, help='coarse-fine-coarse-fine-coarse learning')
    parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)")
    # Misc
    parser.add_argument('--data_path', default='/mnt/sdb1/zhouziyu/ssl/dataset/NIHChestXray/images/images_all', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--imagenet_path', default=None, type=str, help='load imagenet pretrained weights as the initialization.')
    parser.add_argument('--output_dir', default="./pretrained_weight/swinv1_fromIN_unique_multiscale_consis_compdecomp", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=25, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=5, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--cfg',default='./swin_configs/swin_base_img224_window7.yaml', type=str, metavar="FILE", help='path to config file', )
    # parser.add_argument('--cfg',default='./swin_configs/swinv2_base_patch4_window16_512.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if os.path.exists(os.path.join(args.output_dir, "log.txt")):
        log_writer = open(os.path.join(args.output_dir, "log.txt"), 'a')
    else:
        log_writer = open(os.path.join(args.output_dir, "log.txt"), 'w')

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(args)
    dataset = ChestX_ray14(args.data_path,'./data/xray14/official/train_val.txt', augment=transform, data_granularity=1)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.", file=log_writer)

    # ============ building student and teacher networks ... ============
    if 'swinv2_base' in args.arch:
        update_config(config_swinv2, args)
        student = build_model(config_swinv2, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config_swinv2, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        embed_dim = student.num_features

    elif 'swin_base' in args.arch :
        update_config(config, args)

        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        embed_dim = student.num_features

    if args.arch in vits.__dict__.keys():
        # student = vits.__dict__[args.arch](
        #     img_size = [args.image_size],
        #     patch_size=32,
        #     drop_path_rate=args.drop_path_rate,  # stochastic depth
        # )
        # teacher = vits.__dict__[args.arch](img_size = [args.image_size], patch_size=32)
        # embed_dim = student.embed_dim
        student = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/dinov2-base',output_hidden_states=True)
        teacher = AutoModel.from_pretrained('/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/dinov2-base',output_hidden_states=True)
        embed_dim = 768
        
    student = utils.MultiCropWrapper(student, 
                                     UniquenessHead(embed_dim,args.out_dim,use_bn=args.use_bn_in_head,norm_last_layer=args.norm_last_layer),
                                     DenseHead(embed_dim, embed_dim),args)
    teacher = utils.MultiCropWrapper_teacher(teacher, 
                                             UniquenessHead(embed_dim,args.out_dim,use_bn=args.use_bn_in_head,norm_last_layer=args.norm_last_layer),
                                             DenseHead(embed_dim, embed_dim),args)


    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu],find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu],find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(),strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.", file=log_writer)
    log_writer.flush()
    # ============ preparing loss ... ============
    matchingloss = MatchingLoss1to1().cuda()
    uniqueness_loss = UniquenessLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    compdecomp_loss = CompDecompLoss().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.", file=log_writer)
    log_writer.flush()

    # ============ optionally resume training ... ============
    if args.imagenet_path is not None:
        if args.arch == 'swin_base':
            utils.init_from_imagenet(args.imagenet_path, student, teacher)
        elif args.arch == 'swinv2_base':
            utils.init_from_imagenet_swinv2(args.imagenet_path, student, teacher)

    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        matchingloss=matchingloss,
        uniqueness_loss = uniqueness_loss
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !", file=log_writer)
    log_writer.flush()
    for epoch in range(start_epoch, args.epochs):
        dataset = ChestX_ray14(args.data_path,'./data/xray14/official/train_val.txt', augment=transform, data_granularity=args.data_granularity[epoch//30], epoch=epoch)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            )
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp,uniqueness_loss, matchingloss,compdecomp_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, log_writer)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'matchingloss': matchingloss.state_dict(),
            'uniqueness_loss': uniqueness_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), file=log_writer)
    log_writer.flush()


def train_one_epoch(student, teacher, teacher_without_ddp,uniqueness_loss, matchingloss,compdecomp_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, log_writer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    matchingloss.cuda()

    # for it, (images, target_matrix, index1, index2) in enumerate(metric_logger.log_every(data_loader, 50, header, log_writer)):
    for it, data in enumerate(metric_logger.log_every(data_loader, 50, header, log_writer)):
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        loss1, loss2, loss1_decomp, loss2_comp, loss = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()

        if epoch%3==0: # uniqueness
            images = data
            images = [im.cuda(non_blocking=True).float() for im in images]
            # ipdb.set_trace()
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                teacher_cls = teacher(images[:2], epoch)  
                student_cls = student(images, epoch) # global embedding of student, local embeddings of teacher, local embeddings of student

                # ipdb.set_trace()
                loss1 = uniqueness_loss(student_cls, teacher_cls, epoch)
                loss = loss1
        
        elif epoch%3==1: # consistency
            images, target_matrix= data
            images = [im.cuda(non_blocking=True).float() for im in images]
            target_matrix = target_matrix.cuda(non_blocking=True).float()

            # ipdb.set_trace()
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                teacher_spatial = teacher(images, epoch)  
                student_spatial = student(images, epoch) # global embedding of student, local embeddings of teacher, local embeddings of student

                teacher_spatials = teacher_spatial.chunk(2)
                student_spatial = student_spatial.chunk(2)

                # ipdb.set_trace()
                loss2 = matchingloss(student_spatial, teacher_spatials, target_matrix)

                loss = loss2


        else: # comp-decomp
            images,s2lmapping,l2smapping = data
            images = [im.cuda(non_blocking=True).float() for im in images]
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                teacher_spatial = teacher(images, epoch)  
                upsample_features,downsample_features = student(images, epoch) # global embedding of student, local embeddings of teacher, local embeddings of student

                teacher_spatials = teacher_spatial.chunk(2)
                upsample_features = upsample_features.chunk(2)
                downsample_features = downsample_features.chunk(2)


                loss1_decomp, loss2_comp = compdecomp_loss(teacher_spatials,upsample_features,downsample_features,s2lmapping.cuda(),l2smapping.cuda())
                loss3 = (loss1_decomp+loss2_comp)/2 # matrix matching loss
                # loss_vic += loss1_decomp # matrix matching loss
                loss = loss3
            

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                            args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                            args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        #EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for (name_q, param_q), (name_k, param_k) in zip(student.module.named_parameters(), teacher_without_ddp.named_parameters()):
                #print(f"Updating parameter: {name_q} in student, {name_k} in teacher")
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # print(it, loss)
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(uniqueness_loss=loss1.item())
        metric_logger.update(consistency_loss=loss2.item())
        metric_logger.update(comp_loss=loss2_comp.item())
        metric_logger.update(decomp_loss=loss1_decomp.item())


        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class NonLinearConv(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(NonLinearConv, self).__init__()
        self.expand_conv = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=1)
        self.batchnorm_expand = nn.BatchNorm2d(hidden_channels)  # 添加BatchNorm层
        self.activation_expand = nn.ReLU()  # 添加ReLU激活函数
        self.reduce_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1)
        self.batchnorm_reduce = nn.BatchNorm2d(1)  # 添加BatchNorm层
        self.activation_reduce = nn.ReLU()  # 添加ReLU激活函数

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.expand_conv(x)
        # x = self.batchnorm_expand(x)
        x = self.activation_expand(x)
        # x = x.softmax(dim=-1)
        x = self.reduce_conv(x)

        #x = self.batchnorm_reduce(x)
        # x = self.activation_reduce(x)
        x = x.squeeze(1)
        return x

def MLP(mlp, embedding, norm_layer):
    # 修改这里以设置 196 -> 512 -> 196 的结构
    mlp_spec = f"{embedding}-512-{embedding}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    
    for i in range(len(f) - 1):
        layers.append(nn.Linear(f[i], f[i + 1]))
        
        # 如果这不是最后一个线性层，添加规范化和激活
        if i < len(f) - 2:
            if norm_layer == "batch_norm":
                layers.append(nn.BatchNorm1d(f[i + 1]))
            elif norm_layer == "layer_norm":
                layers.append(nn.LayerNorm(f[i + 1]))
            layers.append(nn.ReLU(True))
    
    return nn.Sequential(*layers)

def recall_manual(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN) if TP + FN > 0 else 0


class CompDecompLoss(nn.Module): # compute matrix matching loss
    def __init__(self):
        super(CompDecompLoss, self).__init__()
      
        # Attention Layer
        self.attention = nn.ModuleDict({
            'attention_layer': AttentionLayer()
        })
        #self.loss_view1 = sigmoid_focal_loss(alpha=0.75)#nn.BCEWithLogitsLoss()
        self.loss_view2 = nn.CrossEntropyLoss(ignore_index=-1,reduction='none') #nn.CrossEntropyLoss(ignore_index=-1)
        # MLP Layer

        self.nonlinear =  MLP("512",196,"layer_norm")
        # Loss Criterion
        self.criterion = nn.BCEWithLogitsLoss()


        
    def forward(self, teacher_out, upsample_features,downsample_features,s2lmapping,l2smapping):


        C1_t,C2_t = teacher_out
        C1_s_up, C2_s_up= upsample_features
        C1_s_down, C2_s_down= downsample_features
        # ZA,ZB = teacher_out
        # PA,PB =  student_out_proj
        # logits_A, logits_B = self.attention['attention_layer'](ZA, PB)
        # logits_A_, logits_B_ = self.attention['attention_layer'](PA, ZB)
        logits_A = self.attention['attention_layer'](C2_t, C1_s_up) # logits_A:[B,196,784], logits_B:[B,784,196]
        logits_A_ = self.attention['attention_layer'](C1_t, C2_s_down) # logits_A_:[B,196,49]
        # print(logits_A.shape,logits_B.shape)
        # logits_A = self.nonlinear(logits_A)
        # logits_A_ = self.nonlinear(logits_A_)
        loss_decomp = sigmoid_focal_loss(logits_A,l2smapping.cuda(),alpha=0.99,gamma=0).mean()
        loss_comp = sigmoid_focal_loss(logits_A_,s2lmapping.cuda(),alpha=0.9,gamma=0).mean()


        return loss_decomp,loss_comp 
    

class MatchingLoss1to1(nn.Module): # compute matrix matching loss
    def __init__(self):
        super(MatchingLoss1to1, self).__init__()
      
        # Attention Layer
        self.attention = nn.ModuleDict({
            'attention_layer': AttentionLayer()
        })
        self.criterion = nn.BCEWithLogitsLoss()


        
    def forward(self, student_out, teacher_out, target_matrix):


        ZA,ZB = student_out
        PA,PB =  teacher_out
        logits_A = self.attention['attention_layer'](ZA, PB)
        logits_B = self.attention['attention_layer'](PA, ZB)

        loss1 = sigmoid_focal_loss(logits_A,target_matrix.transpose(-2,-1).cuda(),alpha=0.9,gamma=0).mean()
        loss2 = sigmoid_focal_loss(logits_B,target_matrix.cuda(),alpha=0.9,gamma=0).mean()


        return (loss1+loss2)/2
    
    

# Define the Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, A, B):
        logit_scale = self.logit_scale.exp()
        # logits_A = logit_scale * A @ B.t()
        # logits_B = logits_A.t()
        logits_A = logit_scale * torch.bmm(A, B.transpose(1, 2))
        # logits_B = logits_A.transpose(1, 2)
        return logits_A

# Define the MLP Layer
class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.triplet_loss = InfoNCE(temperature=0.2,negative_mode='unpaired')  # nn.TripletMarginLoss(margin=2, p=2)

    def compute_loss(self, crop1, crop2, bce_labelsl2s, bce_labelss2l):
        """
        input:
         crop1:local embeddings of teacher [B,196,512]
         crop2:local embeddings of student [B,196,512]
         bce_labelsl2s:matrix matching target of large crop to  small crop [B,196,196]
         bce_labelss2l:matrix matching target of small crop to  large crop [B,196,196]
        """
        total_loss = 0

        # Summing over the last dimension
        crop1_index = bce_labelsl2s.sum(dim=2) # [B,196]
        crop2_index = bce_labelss2l.sum(dim=2)

        # Normalizing to get the average
        norm_factor_ori = bce_labelsl2s.sum(dim=1, keepdim=True)
        norm_factor = norm_factor_ori.clone()
        norm_factor[norm_factor == 0] = 1
        norm_factor_expanded = norm_factor.squeeze(1).unsqueeze(-1)  # Change shape from (b, 196) to (b, 196, 1)
        # Calculating average feature
        ##print(norm_factor_expanded.shape)
        average_feature_2_match_1 = torch.bmm(bce_labelsl2s, crop2) / norm_factor_expanded # [B,196,512]

        # Thresholding, find positives
        crop1_index[crop1_index >= 1] = 1
        crop2_index[crop2_index >= 1] = 1
        crop1_index = crop1_index.bool()
        crop2_index = crop2_index.bool()

        # Finding negative indices
        negative_indices1 = torch.where(~crop1_index)
        negative_indices2 = torch.where(~crop2_index)

        # print(f"crop1.shape: {crop1.shape}, crop2.shape: {crop2.shape}")
        # print(f"bce_labelsl2s.shape: {bce_labelsl2s.shape}, bce_labelss2l.shape: {bce_labelss2l.shape}")
        # print(f"average_feature_2_match_1.shape: {average_feature_2_match_1.shape}, crop1_index.shape: {crop1_index.shape}")

        for i in range(crop1.shape[0]):
            if len(negative_indices1[0]) == 0 or len(negative_indices2[0]) == 0 or  crop1[i][crop1_index[i]].shape[0]==0:
                continue
            
            loss = self.triplet_loss(
                crop1[i][crop1_index[i]],  # query
                average_feature_2_match_1[i][crop1_index[i]],  # positive keys
                torch.cat( # negative keys
                    (crop1[i][negative_indices1[1][negative_indices1[0] == i]], 
                     crop2[i][negative_indices2[1][negative_indices2[0] == i]]), 
                    dim=0
                )
            )
            total_loss += loss
        #print(total_loss, crop1.shape[0])
        if isinstance(total_loss, float):
            total_loss = torch.tensor(total_loss, device=crop1.device)
        return total_loss/crop1.shape[0]

    def forward(self, crop1, crop2, bce_labelsl2s,bce_labelss2l):


        return self.compute_loss(crop1, crop2, bce_labelsl2s, bce_labelss2l)


class UniquenessLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)




class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        # if total_loss is float, change it to torch.tensor
        if isinstance(total_loss, float):
            total_loss = torch.tensor(total_loss, device=student_output.device)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        

class DINOLoss_Overlap(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.contrastive_loss = InfoNCE(temperature=0.1,negative_mode='paired')

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_out = F.softmax(student_output / self.student_temp, dim=-1)
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(4)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(4)



        teacher_overlap_crop1 = teacher_out[0] # [B,1024]
        teacher_overlap_crop2 = teacher_out[1] # [B,1024]
        # teacher_nonoverlap_crop1 = teacher_out[2] # [B,1024]
        # teacher_nonoverlap_crop2 = teacher_out[3] # [B,1024]

        student_overlap_crop1 = student_out[0]
        student_overlap_crop2 = student_out[1]
        # student_nonoverlap_crop1 = student_out[2]
        # student_nonoverlap_crop2 = student_out[3]

        # loss1 = self.contrastive_loss(teacher_overlap_crop1, student_overlap_crop2, torch.cat((teacher_nonoverlap_crop1.unsqueeze(1), student_nonoverlap_crop2.unsqueeze(1)), dim=1)) # negatives [B,2,1024]
        # loss2 = self.contrastive_loss(teacher_overlap_crop2, student_overlap_crop1, torch.cat((teacher_nonoverlap_crop2.unsqueeze(1), student_nonoverlap_crop1.unsqueeze(1)), dim=1))
        loss1 = torch.sum(-teacher_overlap_crop2 * F.log_softmax(student_overlap_crop1, dim=-1), dim=-1).mean()
        loss2 = torch.sum(-teacher_overlap_crop1 * F.log_softmax(student_overlap_crop2, dim=-1), dim=-1).mean()
        
        # ipdb.set_trace()
        self.update_center(teacher_output)
        return (loss1+loss2)/2
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
class UniquenessHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=0):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
