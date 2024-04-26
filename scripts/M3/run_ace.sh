
# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 4 --data_path /mnt/dfs/jpang12/datasets/nih_xray14/images/images --output_dir /mnt/dfs/ssiingh/ACE/outputs

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 8 --data_path /Users/shikhharsiingh/JLiangLab/datasets/nih_xray14/images --output_dir /Users/shikhharsiingh/JLiangLab/ACE/outputs/local --saveckp_freq 10 --cfg /scratch/ssiingh/JLiangLab/ACE/swin_configs/swinv2_base_patch4_window16_256.yaml
