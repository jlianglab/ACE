# Benchmark
Benchmark training codes for classification, segmentation and landmark detection tasks. The code can be used to train benchmarks on three backbone: swin-base, vit-base and resnet50 and the backbones can be modified as you like. The segementation and landmark detection tasks architecture are UperNet. These codes are derived from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

## Training 

* For classification task, ddp is added in this code to accelerate the training processing via multiple gpus.
You can use several gpus and set `--nproc_per_node=gpu number`:
```
CUDA_VISIBLE_DEVICES="5,6" python -m torch.distributed.launch --nproc_per_node 2 --master_port=25641 main_cls_ddp.py --img_size 448 --fold 1 --dataset NIHchest
```
Also, you can use one gpu:
```
CUDA_VISIBLE_DEVICES="5" python -m torch.distributed.launch --nproc_per_node 1 --master_port=25641 main_cls_ddp.py --img_size 448 --fold 1 --dataset NIHchest
```

* For segmentation task, ddp is not added. `--local_rank` is used to set device number
```
python main_seg.py --backbone vit_base --pretrain_mode vit_seg_selfpatch --pretrain_weight WEIGHT_PATH --local_rank 6 --dataset SIIM
```

* For landmark detection task, ddp is not added. `--local_rank` is used to set device number
```
python main_keypoint_detect.py --backbone vit_base --pretrain_mode vit_seg_selfpatch --pretrain_weight WEIGHT_PATH --local_rank 6 --dataset SIIM
```


## Testing

* For classification testing, TTA (TenCrop) is used to boost the performance. Use `--resume` to set your testing checkpoint.
```
python test.py --dataset NIHchest --resume WEIGHT_PATH --device 1
```

* For segmentation testing:
```
python test_seg.py --resume WEIGHT_PATH
```

* For landmark testing:
```
python test_keypoint.py --resume WEIGHT_PATH
```

## Some configrations

* Dataloaders
The dataloader file for each dataset which can be found in `.\data`. And I trained four classification datasets: CheXpert, NIHChestXray14, ShenzhenCXR and RSNA Pneumonia, segmentation datasets: JSRT, SIIM, ChestXdet, Montgomery.


* Config

For each dataset, there is a config file locates in `.\configs`. You can modify the initial configrations in these files.

* Load pretrained model

The pretrained model loading function is in `utils.py` file. You may need to change the model keys' name for some pretrained model. Please set `load_checkpoint()` in `utils.py` file.