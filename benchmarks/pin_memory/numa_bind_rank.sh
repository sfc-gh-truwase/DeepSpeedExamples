#!/usr/bin/env bash
# Per-rank NUMA binding for torchrun --no-python launches:
#   torchrun --no-python --nproc_per_node=8 ... ./numa_bind_rank.sh python train.py ...
# Binds the rank's CPUs and preferred memory to its GPU's NUMA node
# (numactl --cpunodebind=N --preferred=N), which also opts the process out of
# automatic NUMA balancing. Refuses to run if the node cannot be determined.
dev="$LOCAL_RANK"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then dev=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f$((LOCAL_RANK + 1))); fi
bdf=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$dev" | tr 'A-F' 'a-f')
node=$(cat "/sys/bus/pci/devices/${bdf: -12}/numa_node" 2>/dev/null || echo -1)
if ! [ "$node" -ge 0 ] 2>/dev/null; then
  echo "[numa] local_rank=$LOCAL_RANK gpu=$dev bdf=$bdf node=unknown; refusing to run unbound" >&2
  exit 3
fi
echo "[numa] local_rank=$LOCAL_RANK gpu=$dev bdf=$bdf node=$node cpunodebind=$node preferred=$node" >&2
exec numactl --cpunodebind="$node" --preferred="$node" "$@"
