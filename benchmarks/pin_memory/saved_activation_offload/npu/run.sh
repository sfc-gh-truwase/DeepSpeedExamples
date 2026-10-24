#!/bin/bash
# Saved-activation offload — Ascend 910B4 one-click run
# Parameters pinned for this card (29.5 GiB usable):
#   Qwen3-4B / batch 1 x seq 2048 / K=3 / --credits 1 / ZeRO-2 CPU AdamW
# Usage:
#   ./run.sh                          # model from the HF hub
#   MODEL=/path/to/Qwen3-4B ./run.sh  # local checkpoint
# Results and logs go to ./results/.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Pin OpenMP threads to the container CPU quota (42 cores on this host);
# comment out on non-container setups.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
ulimit -l unlimited 2>/dev/null || true

# conda environment and CANN toolkit; skipped when already activated.
# Override CONDA_ENV / CONDA_SH / CANN_SET_ENV for different setups.
CONDA_ENV="${CONDA_ENV:-cann}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CANN_SET_ENV="${CANN_SET_ENV:-$HOME/Ascend/cann-8.5.1/set_env.sh}"
if [ -f "$CONDA_SH" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi
if [ -f "$CANN_SET_ENV" ]; then
    # shellcheck disable=SC1091
    source "$CANN_SET_ENV"
fi

mkdir -p results
TS="$(date +%F_%H%M%S)"

# --- NUMA launch ---
# Place host memory on the NPU's NUMA node (set NUMA_NODE=<n>; read it from
# /sys/bus/pci/devices/<NPU BDF>/numa_node) and keep CPU threads there. On H200,
# remote host memory cost +1.1-1.5 s/step (CPU Adam). An explicit policy also opts
# the process out of automatic NUMA balancing (kernel.numa_balancing=1), whose
# hinting faults hit every large host buffer each step. --preferred spills to other
# nodes instead of failing if one node is too small. If the node is unknown, fall
# back to --membind=all (balancing off, placement unrestricted). NUMA_NODE=<n>
# overrides detection; NUMA_BIND=0 disables. See ../../README.md.
LAUNCH=()
if [ "${NUMA_BIND:-1}" = 1 ] && command -v numactl >/dev/null 2>&1; then
    NODE="${NUMA_NODE:-}"
    if [ -n "$NODE" ] && [ "$NODE" -ge 0 ] 2>/dev/null; then
        LAUNCH=(numactl --cpunodebind="$NODE" --preferred="$NODE")
    elif [ "$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo 0)" != 0 ]; then
        LAUNCH=(numactl --membind=all)
    fi
fi
echo "launch: ${LAUNCH[*]:-(none)}"
# --- end NUMA launch ---

${LAUNCH[@]+"${LAUNCH[@]}"} python ../bench.py --model "$MODEL" --batch 1 --seq 2048 \
    --offload-tensors 3 --offload-optimizer --act-mem-opt \
    --steps 20 --warmup 5 --credits 1 \
    --manifest-out results/qwen3-4b-seq2048-k3-manifest.json \
    2>&1 | tee "results/saved_activation_${TS}.log"
