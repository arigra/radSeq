#!/bin/bash
#SBATCH --job-name=radseq_step6
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_step6_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_step6_%j.err
# Step 6 (new vehicle model), resuming wherever it stopped. Safe to resubmit.
# If another copy is still running (e.g. in an interactive session, on any
# node), this waits for its heartbeat to go stale, then takes over.
# Submit from the login host ece-hpc:  sbatch scripts/step6.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
LOCK=logs/step6.heartbeat
while [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; do sleep 120; done
bash scripts/run_step6.sh
