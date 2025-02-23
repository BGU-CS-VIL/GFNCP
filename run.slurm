#!/bin/bash

#SBATCH --partition main  ### choose from: [--partition rtx2080, rtx3090, rtx6000, main]
#SBATCH --time 7-00:00:00
#SBATCH --job-name 'GFN_main'
#SBATCH --output log.log
#SBATCH --gpus=rtx_3090:1    ### choose from: [rtx_3090:1, gpus=rtx_2080:1, gpus=rtx_6000:1]. Here we can request multiple gpus, e.g.: --gpus=rtx_3090:4
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
##SBATCH --nodelist=cs-3090-04
##SBATCH --qos=orenfr  ### choose: [pakman, orenfr]

echo 'date'
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

module load anaconda

source activate iritcpab4

conda env list

# scl enable devtoolset-9 bash
# export LD_LIBRARY_PATH=/home/tohamy/.conda/envs/iritcpab4/lib:$LD_LIBRARY_PATH
# export PATH=/opt/rh/devtoolset-9/root/usr/bin/:$PATH

# export CC=/home/tohamy/.conda/envs/iritcpab4/x86_64-conda-linux-gnu
# export CXX=/home/wmcneill/anaconda3/bin/x86_64-conda-linux-gnu-c++
# export GCC=/home/tohamy/.conda/envs/iritcpab4/x86_64-conda-linux-gnu
# export GXX=/home/wmcneill/anaconda3/bin/x86_64-conda-linux-gnu-g++

# ------ RUNNING LINE ---------
# --load-model
# --run-geweke
# --dataset choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR', 'IN50_ftrs', 'IN100_ftrs', 'IN200_ftrs', 'CIFAR_ftrs', 'tinyimagenet']
# --eval-best-model  # to compute metrics after the model has converged

# python -u main.py --dataset CIFAR_ftrs --show-histogram --load-model --run-geweke --experiment test_run
python -u main.py --dataset MNIST --show-histogram --eval-best-model --experiment test_main_code