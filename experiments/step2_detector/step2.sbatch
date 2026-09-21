#!/bin/bash
#SBATCH --job-name=radseq_step2
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_step2_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_step2_%j.err
# Continue (or start) research plan step 2 from wherever it stopped.
# Submit from the login host ece-hpc:  sbatch experiments/step2_detector/step2.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash experiments/step2_detector/run_step2.sh
