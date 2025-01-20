#! /usr/bin/env bash

#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

module load gcc/13.2.0-iqpfkya
module load cuda/12.4.0-zk32gam

echo "Job started at $(date)"
echo "Running on node: $SLURM_NODELIST"

build/cuda_bitonic_sort 20

echo "Job finished at $(date)"
