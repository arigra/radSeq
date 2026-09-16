#!/bin/bash
#SBATCH --job-name=radseq_e3_full
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_e3_full_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_e3_full_%j.err
# Continue (or start) the joint-attention full-data run from its last checkpoint.
# Submit from the login host ece-hpc:  sbatch scripts/e3_full.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_e3_full.sh
