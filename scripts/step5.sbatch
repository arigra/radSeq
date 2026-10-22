#!/bin/bash
#SBATCH --job-name=radseq_step5
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_step5_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_step5_%j.err
# Step 5, resuming wherever it stopped. Safe to resubmit; if an interactive
# copy is still running, run_step5.sh refuses to start a second.
# Submit from the login host ece-hpc:  sbatch scripts/step5.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_step5.sh
