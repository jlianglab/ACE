#!/bin/bash
#SBATCH --job-name=ace-nih-pretrain
#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH -t 4-0:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -G a100:4
#SBATCH -o %x.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e %x.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=ssiingh@asu.edu

#SBATCH --export=NONE   # Purge the job-submitting shell environment
module load mamba/latest
source activate ml10
cd /scratch/ssiingh/JLiangLab/ACE
torchrun --nproc_per_node=4 main.py --arch swin_base --batch_size_per_gpu 8 --data_path /data/jliang12/shared/dataset/NIHCXR14/full_images/images --output_dir /scratch/ssiingh/JLiangLab/ACE/outputs
