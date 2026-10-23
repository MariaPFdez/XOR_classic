#!/usr/bin/env bash
# Launch the grid on several DGX nodes that see this directory at the same
# path over NFS.  Each node runs PER_HOST shard processes (the GB10 is mostly
# idle with these small models, so 2 per node is a good default).
#
#   ./launch_shards.sh spark1 spark2 spark3
#
# Environment:
#   XORLAB_DATA   shared dataset dir           (default: $DIR/data)
#   TORCH_HOME    shared torch cache (LPIPS)   (default: $DIR/torch_home)
#   PER_HOST      shard processes per node     (default: 2)
#   RUNNER        venv | docker                (default: venv)
#   SEEDS         seed range                   (default: 0-4)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
HOSTS=("$@"); [ ${#HOSTS[@]} -gt 0 ] || { echo "usage: $0 host1 [host2 ...]"; exit 1; }
PER_HOST="${PER_HOST:-2}"; RUNNER="${RUNNER:-venv}"; SEEDS="${SEEDS:-0-4}"
XORLAB_DATA="${XORLAB_DATA:-$DIR/data}"; TORCH_HOME="${TORCH_HOME:-$DIR/torch_home}"
N=$(( ${#HOSTS[@]} * PER_HOST ))
mkdir -p "$DIR/logs" "$DIR/results"
k=0
for h in "${HOSTS[@]}"; do
  for _ in $(seq 1 "$PER_HOST"); do
    if [ "$RUNNER" = docker ]; then
      PY="docker run --rm --gpus all --ipc=host -u $(id -u):$(id -g) -v $DIR:$DIR -v $XORLAB_DATA:$XORLAB_DATA -v $TORCH_HOME:$TORCH_HOME -e XORLAB_DATA=$XORLAB_DATA -e TORCH_HOME=$TORCH_HOME -w $DIR xorlab python"
    else
      PY="$DIR/.venv/bin/python"
    fi
    CMD="cd $DIR && export XORLAB_DATA=$XORLAB_DATA TORCH_HOME=$TORCH_HOME && nohup $PY grid.py --seeds $SEEDS --shard $k --nshards $N > logs/shard_${k}_of_${N}_${h}.log 2>&1 &"
    echo "[$h] shard $k/$N"
    ssh "$h" "$CMD"
    k=$((k+1))
  done
done
echo "launched $N shards. Monitor:  tail -f $DIR/logs/*.log    Progress:  ls $DIR/results/*/seed*.json | wc -l"
