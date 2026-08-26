#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=2
#SBATCH --time=1-12:00
#SBATCH --partition=gpu_a100   # change to your partition

bash "$1"
