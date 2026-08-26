#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=160GB
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00
#SBATCH --partition=gpu_a100   # change to your partition

bash "$1"
