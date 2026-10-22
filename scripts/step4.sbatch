#!/bin/bash
#SBATCH --job-name=radseq_step4
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_step4_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_step4_%j.err
# Step 4 (fine-tune, render, detector arms, both variants), then the
# memorisation check, each resuming where it stopped. Safe to resubmit; if an
# interactive copy is still running, run_step4.sh refuses to start a second.
# Submit from the login host ece-hpc:  sbatch scripts/step4.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_step4.sh && bash scripts/run_memcheck_scene.sh
