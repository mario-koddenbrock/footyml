#!/bin/bash
#SBATCH --job-name=footyml_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=high
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=.logs/%A_%x.log
#SBATCH --error=.logs/%A_%x.log
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job name:     $SLURM_JOB_NAME"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE"
echo "Started:      $(date)"
echo "============================================================"

source ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"   # ensure uv is on PATH

cd "$HOME/workspace/footyml"

echo "==> Installing dependencies with uv ..."
uv sync --extra dev

if [ $? -ne 0 ]; then
    echo "ERROR: uv sync failed"
    exit 1
fi

uv run python --version
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
                  print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

set -e
mkdir -p .logs data/models data/predictions

# ---------------------------------------------------------------------------
# Step 1 — Train TabPFN on all leagues (GPU, no sample limit)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Training ---"
uv run python scripts/train.py \
    --league all \
    --season 2025 \
    --train-from 2015 \
    --eval \
    --device cuda

# ---------------------------------------------------------------------------
# Step 2 — Predict WC 2026 and export to JSON
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: WC 2026 predictions ---"
uv run python scripts/predict_tournament.py \
    --competition WC \
    --season 2026 \
    --export data/predictions/wc2026_upcoming

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Finished at $(date)"
echo "Output files:"
ls -lh data/models/
echo "============================================================"
