#!/bin/zsh
# Sync FootyML data and code to/from the HPC cluster.
#
# Usage:
#   ./scripts/cluster_sync.sh push   # push data + code to cluster, then submit job
#   ./scripts/cluster_sync.sh pull   # pull model + predictions back
#   ./scripts/cluster_sync.sh status # check job status

CLUSTER="cluster"
REMOTE_DIR="~/workspace/footyml"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

case "$1" in

  push)
    echo "==> Syncing code and data to $CLUSTER:$REMOTE_DIR ..."
    rsync -avz --progress \
      --exclude='.venv/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.git/' \
      --exclude='data/raw/' \
      "$LOCAL_DIR/" "$CLUSTER:$REMOTE_DIR/"

    echo ""
    echo "==> Installing dependencies on cluster (uv) ..."
    ssh "$CLUSTER" "
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      cd $REMOTE_DIR && uv sync --extra dev 2>&1 | tail -5
    "

    echo ""
    echo "==> Submitting SLURM job ..."
    ssh "$CLUSTER" "cd $REMOTE_DIR && sbatch scripts/train_hpc.sbatch"
    ;;

  pull)
    echo "==> Pulling model and predictions from $CLUSTER:$REMOTE_DIR ..."
    rsync -avz --progress \
      "$CLUSTER:$REMOTE_DIR/data/models/" "$LOCAL_DIR/data/models/"
    rsync -avz --progress \
      "$CLUSTER:$REMOTE_DIR/data/predictions/" "$LOCAL_DIR/data/predictions/"
    rsync -avz --progress \
      "$CLUSTER:$REMOTE_DIR/.logs/" "$LOCAL_DIR/.logs/"
    echo ""
    echo "==> Done. Files pulled:"
    ls -lh "$LOCAL_DIR/data/models/" "$LOCAL_DIR/data/predictions/" 2>/dev/null
    ;;

  status)
    echo "==> SLURM jobs on $CLUSTER:"
    ssh "$CLUSTER" "squeue -u \$USER --format='%.18i %.9P %.20j %.8T %.10M %.6D %R'"
    echo ""
    echo "==> Recent log tail:"
    ssh "$CLUSTER" "ls -t $REMOTE_DIR/.logs/*.log 2>/dev/null | head -1 | xargs tail -20 2>/dev/null || echo 'no logs yet'"
    ;;

  *)
    echo "Usage: $0 {push|pull|status}"
    echo ""
    echo "  push   — rsync repo to cluster and submit sbatch job"
    echo "  pull   — rsync model + predictions back to local"
    echo "  status — check SLURM queue and tail latest log"
    exit 1
    ;;
esac
