#!/bin/bash
# FILENAME:  Ark-SPECIFIC-ChestXrya14-finetune_1

#SBATCH -A med220025-gpu       # allocation name
#SBATCH --nodes=1            # Total # of nodes 
#SBATCH --ntasks-per-node=8   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=4     # Number of GPUs per node
#SBATCH --time=2-0:00:00     # Total run time limit (hh:mm:ss)
#SBATCH --mem=80           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -J ark-spec-base-chestxray14-finetune_1        # Job name
#SBATCH -o %x_myjob.o%j          # Name of stdout output file
#SBATCH -e %x_myjob.e%j          # Name of stderr error file
#SBATCH -p gpu                # Queue (partition) name
#SBATCH --mail-user=ssiingh@asu.edu
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
module module load anaconda/2021.05-py38
source activate ace
cd /anvil/scratch/x-ssiingh/JLiangLab/ACE

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --master_port 28301 main.py --arch swin_base --batch_size_per_gpu 16 --data_path /anvil/scratch/x-ssiingh/JLiangLab/datasets/nih_xray14/nih_xray14/images/images --output_dir /anvil/scratch/x-ssiingh/JLiangLab/ACE/outputs/local_comp_decomp --saveckp_freq 1 --cfg /anvil/scratch/x-ssiingh/JLiangLab/ACE/swin_configs/swin_base_img224_window7.yaml
