#!/bin/bash
#SBATCH --job-name=radseq_e3_standard
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_e3_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_e3_%j.err
# Continue the full-data standard-recipe run (resumes from the last checkpoint)
# and score it. Submit from the login host: sbatch scripts/e3_standard.sbatch

cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_e3_standard.sh
