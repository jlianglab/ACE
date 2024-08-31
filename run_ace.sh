# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 4 --data_path /mnt/dfs/jpang12/datasets/nih_xray14/images/images --output_dir /mnt/dfs/ssiingh/ACE/outputs

#!/bin/bash
#SBATCH --job-name=ace-nih-pretrain
#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH -t 7-0:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -G a100:1
#SBATCH -o %x.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e %x.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=ssiingh@asu.edu

#SBATCH --export=NONE   # Purge the job-submitting shell environment
module load mamba/latest
source activate ace
cd /scratch/ssiingh/JLiangLab/ACE
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 16 --data_path /data/jliang12/shared/dataset/NIHCXR14/full_images/images --output_dir /scratch/ssiingh/JLiangLab/ACE/outputs/fundus --saveckp_freq 1 --cfg /scratch/ssiingh/JLiangLab/ACE/swin_configs/swin_base_img224_window7.yaml