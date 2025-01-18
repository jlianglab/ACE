# ACE: Anatomically Consistency Embedding via Composition and Decomposition

# Pretrain ACE models:

Using DDP to pretrain ACE:
```
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 8 --data_path pretrain image path --output_dir 
```


Pretrained weights:

| Model name | Backbone | Input Resolution | model |
|------------|----------|------------------|-------|
|PEAC | Swin-B-v1 | 448x448 | [download](https://drive.google.com/drive/folders/1xHKDWPQbMw7D6mZRLLWNSTgLH79lI2sl?usp=sharing)| 