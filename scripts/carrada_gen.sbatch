#!/bin/bash
#SBATCH --job-name=radseq_carrada
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_carrada_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_carrada_%j.err
# Step 8 (docs/notes/2026-10-24-carrada_gen-generator-realism.md), resuming wherever it
# stopped. Safe to resubmit. If another copy is still running (e.g. in an
# interactive session, on any node), this waits for its heartbeat to go stale.
# Submit from the login host ece-hpc:  sbatch scripts/carrada_gen.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
LOCK=logs/carrada_gen.heartbeat
while [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 300 ]; do sleep 120; done
bash scripts/run_carrada_gen.sh
