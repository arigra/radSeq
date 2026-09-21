#!/bin/bash
#SBATCH --job-name=radseq_pretrain
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_pretrain_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_pretrain_%j.err
# Step 3 pretraining on both scene-simulator variants, resuming from the last
# checkpoint. Safe to resubmit. Do not submit while an interactive session is
# running scripts/run_pretrain.sh (it refuses to start a second copy anyway).
# Submit from the login host ece-hpc:  sbatch scripts/pretrain.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_pretrain.sh
