#!/bin/bash
#SBATCH --job-name=radseq_fidelity
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_fidelity_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_fidelity_%j.err
# Continue (or start) the sim-to-sim fidelity pilot from its last completed
# stage. Safe to submit repeatedly; finished stages are skipped.
# Submit from the login host ece-hpc:  sbatch scripts/fidelity_pilot.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_fidelity_pilot.sh
