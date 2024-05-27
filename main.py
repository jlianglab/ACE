

# CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port 28300 main.py --arch swin_base --batch_size_per_gpu 8


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
from config import update_config
from config import save_config
from models import build_model
from transforms import MultiCropTrainDataTransform, DataAugmentationDINO
from losses import globalconsis_loss
from einops import rearrange
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import recall_score
from torch import autograd
import ipdb
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'cvt_small', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'vil_14121', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
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

    # optional training components
    parser.add_argument('--global_loss', default=False, type=utils.bool_flag, help="Whether to use global loss")
    parser.add_argument('--matrix_matching', default=True, type=utils.bool_flag, help="Whether to use matrix matching loss")
    parser.add_argument('--comp', default=False, type=utils.bool_flag, help="Whether to composition and decomposition loss")

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
    parser.add_argument('--epochs', default=302, type=int, help='Number of epochs of training.')
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
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
        help="Whether to use dense prediction in projection head (Default: False)")
    # Misc
    parser.add_argument('--data_path', default='/data1/zhouziyu/liang/NIHChestXray/images/images_all/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/ssd2/zhouziyu/ssl/github/ACE/pretrained_weight/from_imagenet_weighted_matrix", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--cfg',default='./swin_configs/swin_base_img224_window7.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser


def train_dino(args):
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
    transform = DataAugmentationDINO()
    #transform =DataAugmentationDINO()
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    #dataset = ImageFolder_vindr(args.data_path, transform=transform)
    # dataset = LDPolyp(augment=transform)
    dataset = ChestX_ray14(args.data_path,'./data/xray14/official/train_val.txt', augment=transform)
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
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    if 'swin' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.num_features,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)
    
    student = build_model(config, use_dense_prediction=args.use_dense_prediction)
    teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
    embed_dim = student.num_features
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ),DenseHead(),args)
    teacher = utils.MultiCropWrapper_teacher(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        DenseHead(), args
    )


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
    barlow_loss = AttentionMLPModel(512,512,1).cuda()#BarlowLoss(
    # dino_loss = DINOLoss(
    #     args.out_dim,
    #     2,  # total number of crops
    #     args.warmup_teacher_temp,
    #     args.teacher_temp,
    #     args.warmup_teacher_temp_epochs,
    #     args.epochs,
    # ).cuda()
    dino_loss = ContrastiveLoss().cuda()

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
    utils.init_from_imagenet('/ssd2/zhouziyu/ssl/contrast_12N/pretrained_weight/ldpolyp/swin_base_patch4_window7_224_imagenet1k.pth',student, teacher)

    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        barlow_loss=barlow_loss,
        dino_loss = dino_loss
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !", file=log_writer)
    log_writer.flush()
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp,dino_loss, barlow_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, log_writer)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'barlow_loss': barlow_loss.state_dict(),
            'dino_loss': dino_loss.state_dict(),
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


def train_one_epoch(student, teacher, teacher_without_ddp,dino_loss, barlow_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, log_writer):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss =nn.MSELoss()
    barlow_loss.cuda()
    criterion = nn.CosineSimilarity(dim=1).cuda()
    # triplet_loss =barlow_loss.cuda()
    local_loss = TripletLoss()
    #torch.autograd.set_detect_anomaly(True)
    for it, ((images,local_crops,locations,s2lmapping,l2smapping, sample_index1, sample_index2), _) in enumerate(metric_logger.log_every(data_loader, 50, header, log_writer)):
        # locations:the overlap mask of two crops(14*14), s2lmapping:matrix matching target(196*196)
        # ipdb.set_trace()
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True).float() for im in images]
        local_crops = [im.cuda(non_blocking=True).float() for im in local_crops]
        sample_index = torch.cat([sample_index1.cuda(non_blocking=True), sample_index2.cuda(non_blocking=True)])

        # locations=[location.cuda(non_blocking=True) for location in locations]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # with autograd.detect_anomaly():
            teacher_cls, teacher_spatial, teacher_comp, teacher_decomp = teacher(images, local_crops, sample_index)  
            student_cls, student_spatial, student_spatial_pred, student_comp, student_decomp = student(images, local_crops, sample_index) # global embedding of student, local embeddings of teacher, local embeddings of student

            student_spatials_pred = student_spatial_pred.chunk(2)
            teacher_spatials = teacher_spatial.chunk(2)


            loss_vic=0
            loss_decomp=0
            loss = 0
            
            # global_loss = dino_loss(student_cls, teacher_cls, epoch)

            if args.global_loss:
                global_loss = dino_loss(teacher_cls, student_cls)
            if args.matrix_matching:
                loss1, loss2 = barlow_loss(teacher_spatials,student_spatials_pred,locations[0],locations[1],s2lmapping.cuda(),l2smapping.cuda())
                loss_vic += (loss1+loss2)/2 # matrix matching loss
                loss = loss+loss_vic

            # loss_local =  local_loss(student_spatials[0], student_spatials_pred[1], l2smapping.cuda(),s2lmapping.cuda()) + local_loss(student_spatials_pred[0], student_spatials[1], l2smapping.cuda(),s2lmapping.cuda()) #contrastive learning loss
            if args.comp:
                loss_comp = F.smooth_l1_loss(student_comp, teacher_comp)
                
                # # ipdb.set_trace()
                for i in range(4):
                    loss_decomp+=F.smooth_l1_loss(student_decomp[i], teacher_decomp[i])
                loss_decomp = loss_decomp/4

            # loss = (loss_vic+global_loss+loss_comp+loss_decomp)/4 # 
            # loss = global_loss
            # loss = loss_local

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

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        # metric_logger.update(order_loss=order_loss.item())
        if args.global_loss:
            metric_logger.update(global_loss=global_loss.item())
        if args.matrix_matching:
            metric_logger.update(loss_matrix=loss_vic.item())
        if args.comp:
            metric_logger.update(loss_comp=loss_comp.item())
            metric_logger.update(loss_decomp=loss_decomp.item())
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


class AttentionMLPModel(nn.Module): # compute matrix matching loss
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionMLPModel, self).__init__()
      
        # Attention Layer
        self.attention = nn.ModuleDict({
            'attention_layer': AttentionLayer()
        })
        #self.loss_view1 = sigmoid_focal_loss(alpha=0.75)#nn.BCEWithLogitsLoss()
        self.loss_view2 = nn.CrossEntropyLoss(ignore_index=-1,reduction='none') #nn.CrossEntropyLoss(ignore_index=-1)
        # MLP Layer
        self.mlp = nn.ModuleDict({
            'mlp_layer': MLPLayer(input_dim, hidden_dim, output_dim)
        })
        self.nonlinear =  MLP("512",196,"layer_norm")
        # Loss Criterion
        self.criterion = nn.BCEWithLogitsLoss()


        
    def forward(self, student_out, student_out_proj, locations0,locations1,s2lmapping,l2smapping):


        ZA,ZB = student_out
        PA,PB =  student_out_proj
        logits_A, logits_B = self.attention['attention_layer'](ZA, PB)
        logits_A_, logits_B_ = self.attention['attention_layer'](PA, ZB)
        # print(logits_A.shape,logits_B.shape)
        # logits_A = self.nonlinear(logits_A)
        # logits_A_ = self.nonlinear(logits_A_)
        loss1 = ( sigmoid_focal_loss(logits_A,l2smapping.cuda(),alpha=0.9,gamma=0).mean()+ sigmoid_focal_loss(logits_A_,l2smapping.cuda(),alpha=0.9,gamma=0).mean())/2
        loss2 = ( sigmoid_focal_loss(logits_B,s2lmapping.cuda(),alpha=0.9,gamma=0).mean()+ sigmoid_focal_loss(logits_B_,s2lmapping.cuda(),alpha=0.9,gamma=0).mean())/2


        # # Get predicted labels
        # predicted_A = (torch.sigmoid(logits_A) >= 0.5).float()
        # predicted_B = (torch.sigmoid(logits_B) >= 0.5).float()





        return loss1,loss2

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
        logits_B = logits_A.transpose(1, 2)
        return logits_A, logits_B

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


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.contrastive_loss = InfoNCE(temperature=0.1,negative_mode='paired')

    def forward(self, teacher_cls, student_cls):
        teacher_cls = teacher_cls.chunk(4)
        student_cls = student_cls.chunk(4)

        teacher_overlap_crop1 = teacher_cls[0] # [B,1024]
        teacher_overlap_crop2 = teacher_cls[1] # [B,1024]
        teacher_nonoverlap_crop1 = teacher_cls[2] # [B,1024]
        teacher_nonoverlap_crop2 = teacher_cls[3] # [B,1024]

        student_overlap_crop1 = student_cls[0]
        student_overlap_crop2 = student_cls[1]
        student_nonoverlap_crop1 = student_cls[2]
        student_nonoverlap_crop2 = student_cls[3]

        loss1 = self.contrastive_loss(teacher_overlap_crop1, student_overlap_crop2, torch.cat((teacher_nonoverlap_crop1.unsqueeze(1), student_nonoverlap_crop2.unsqueeze(1)), dim=1)) # negatives [B,2,1024]
        loss2 = self.contrastive_loss(teacher_overlap_crop2, student_overlap_crop1, torch.cat((teacher_nonoverlap_crop2.unsqueeze(1), student_nonoverlap_crop1.unsqueeze(1)), dim=1))

        return (loss1+loss2)/2

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
