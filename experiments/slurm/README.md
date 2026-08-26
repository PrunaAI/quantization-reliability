# SLURM submission templates

These are example SLURM batch scripts for the TUM gpu_gtx_1080 and gpu_a100 partitions.
Adapt partition names, memory, and GPU counts for your cluster.

## Usage
Run the Hydra sweep script inside a SLURM job:

    sbatch experiments/slurm/single_gpu_job.sh experiments/sweeps/llama32_1b_triviaqa.sh
