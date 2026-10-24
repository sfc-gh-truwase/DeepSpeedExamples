#!/bin/bash
# Activation checkpoint offload (recipe C) — H200 one-click run
# Parameters pinned for this card (~139.8 GiB usable):
#   Qwen3-8B / batch 1 x seq 32768 / ZeRO-2 CPU AdamW / Liger fused linear CE
#   + TiledMLP / activation offload (CheckpointHiddenStatesOffload)
#   / --steps 10 --warmup 1
# Usage:
#   ./run.sh                          # model from the HF hub (config only, random weights)
#   MODEL=/path/to/Qwen3-8B ./run.sh  # local config
# Results and logs go to ./results/.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -l unlimited 2>/dev/null || true

# $REPO is the sfc_ds_pin_memory workspace root, five levels above this
# script; it must be on PYTHONPATH for the benchmark's own package imports.
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# conda environment (torch 2.14 + C++20 flash-attn, see ../README.md);
# skipped when already activated. Override CONDA_ENV / CONDA_SH.
CONDA_ENV="${CONDA_ENV:-torch214}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

mkdir -p results
TS="$(date +%F_%H%M%S)"

# --- NUMA launch ---
# Place host memory on the GPU's NUMA node and keep CPU threads there. On H200,
# remote host memory cost +1.1-1.5 s/step (CPU Adam). An explicit policy also opts
# the process out of automatic NUMA balancing (kernel.numa_balancing=1), whose
# hinting faults hit every large host buffer each step. --preferred spills to other
# nodes instead of failing if one node is too small. If the node is unknown, fall
# back to --membind=all (balancing off, placement unrestricted). NUMA_NODE=<n>
# overrides detection; NUMA_BIND=0 disables. See ../../README.md.
LAUNCH=()
if [ "${NUMA_BIND:-1}" = 1 ] && command -v numactl >/dev/null 2>&1; then
    NODE="${NUMA_NODE:-}"
    if [ -z "$NODE" ] && command -v nvidia-smi >/dev/null 2>&1; then
        BDF="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null \
            | tr 'A-F' 'a-f')"
        NODE="$(cat "/sys/bus/pci/devices/${BDF: -12}/numa_node" 2>/dev/null || echo -1)"
    fi
    if [ -n "$NODE" ] && [ "$NODE" -ge 0 ] 2>/dev/null; then
        LAUNCH=(numactl --cpunodebind="$NODE" --preferred="$NODE")
    elif [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo 0)" != 0 ]; then
        LAUNCH=(numactl --membind=all)
    fi
fi
echo "launch: ${LAUNCH[*]:-(none)}"
# --- end NUMA launch ---

${LAUNCH[@]+"${LAUNCH[@]}"} python ../bench.py --model "$MODEL" --batch 1 --seq 32768 \
    --offload-optimizer --act-mem-opt \
    --steps 10 --warmup 1 \
    2>&1 | tee "results/activation_offload_${TS}.log"
