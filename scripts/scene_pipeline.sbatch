#!/bin/bash
#SBATCH --job-name=radseq_scene
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_scene_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_scene_%j.err
# Scene-simulator caches, then the 512x256 recipe check. Both resume from
# their last finished chunk/checkpoint, so resubmitting is always safe.
# Do not run while an interactive session is running the same scripts.
# Submit from the login host ece-hpc:  sbatch scripts/scene_pipeline.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash scripts/run_scene_caches.sh &
bash scripts/run_scene_smoke.sh
wait
