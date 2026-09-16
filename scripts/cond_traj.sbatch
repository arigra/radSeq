#!/bin/bash
#SBATCH --job-name=radseq_cond_traj
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_cond_traj_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_cond_traj_%j.err
# Continue (or start) the trajectory-conditioned run from its last checkpoint.
# Submit from the login host ece-hpc:  sbatch scripts/cond_traj.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_cond_traj.sh
