#!/bin/bash
#SBATCH --job-name=radseq_e3_long
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_e3_long_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_e3_long_%j.err
# Continue (or start) the longer-training full-data run from its last checkpoint.
# Submit from the login host ece-hpc:  sbatch scripts/e3_long.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_e3_long.sh
