#!/bin/bash
#SBATCH --job-name=radseq_step2b
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_step2b_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_step2b_%j.err
# Continue (or start) research plan step 2b (memorisation, synthetic data, class detector) from wherever it stopped.
# Submit from the login host ece-hpc:  sbatch scripts/step2b.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_step2b.sh
