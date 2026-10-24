# Pinned-memory experiments

Harnesses for pin vs pageable host memory on CPU offload. Each subdirectory is
one experiment. Shared subprocess/JSON helpers live in `common.py`.

Native backends `mlock` host memory: raise `RLIMIT_MEMLOCK` (`ulimit -l`) or
run as root for multi-GB models.

## NUMA balancing

On multi-socket hosts with `kernel.numa_balancing=1` (H200 nodes: 2 nodes, 192 CPUs), the
kernel periodically unmaps the pages of large, long-lived host buffers to sample access, and
migrates them. On 4 KiB pages this costs 1.6–6.8M minor faults per step (exactly
`numa_hint_faults` + `numa_pages_migrated`), in **both** arms. They land in backward and the
CPU Adam step, along with 10–70 CPU-s of sys time, and make step times noisy. Host memory **placement** matters too:
on H200, host memory on the node remote from the GPU cost +1.1-1.5 s/step, mostly in the CPU Adam
step. Pinned DMA bandwidth is the same either way; pageable H2D drops 27%. So the cuda `run.sh`
scripts launch under `numactl --cpunodebind=<gpu node> --preferred=<gpu node>`, with the node read from
sysfs for the first visible GPU. The npu scripts need `NUMA_NODE=<n>`. Without a known node, they
fall back to `numactl --membind=all`, which disables balancing but leaves placement unrestricted.
Either explicit policy opts the process out of NUMA balancing. When calling `bench.py` directly:

```bash
N=$(cat /sys/bus/pci/devices/<gpu BDF>/numa_node)
numactl --cpunodebind=$N --preferred=$N python model_tensor_offload/bench.py --model Qwen/Qwen3-8B --zero-stage 3
```

For multi-rank runs, bind each rank to its own device's node, never all ranks to one node.
Arm subprocesses inherit the policy (check: `/proc/<pid>/numa_maps` shows `prefer:<n>` or `bind:`).
`NUMA_BIND=0 ./run.sh` measures with no explicit policy (balancing on, if enabled). Set `PIN_BENCH_STEP_PROBE=<dir>` to log
per-step/per-phase faults, CPU time and NUMA counters (`StepProbe` in `common.py`).

## Layout

| Folder | Blog experiment | Default command |
|--------|-----------------|-----------------|
| [`model_tensor_offload/`](model_tensor_offload/) | ZeRO CPU param/optimizer offload (stage 3 default; `--zero-stage 1\|2` optional) | `python model_tensor_offload/bench.py --hidden 2048 --layers 12 --batch 4 --seq 128` |
| [`activation_offload/`](activation_offload/) | Checkpoint hidden-state offload; `use_pin_memory` on/off, **async on** | `python activation_offload/bench.py --hidden 1024 --layers 8 --batch 1 --seq 2048` |
| [`saved_activation_offload/`](saved_activation_offload/) | Credit-bounded saved MLP activation offload (no remat) | `python saved_activation_offload/bench.py --model Qwen/Qwen3-8B --offload-optimizer` |
| [`h2d_d2h/`](h2d_d2h/) | Supporting H2D/D2H GB/s (pageable, torch, native-unregistered, native-registered) | `python h2d_d2h/bench.py` |
| [`grad_offload/`](grad_offload/) | Optional #8207-style grad offload (wraps model-tensor ZeRO-3) | `python grad_offload/bench.py --hidden 2048 --layers 12` |
| [`cpu_pin/`](cpu_pin/) | Optional CPU-only native vs Torch pin | `python cpu_pin/bench.py` |
| [`deepcompile_activation/`](deepcompile_activation/) | Optional `compile.offload_activation_pin_memory` | `python deepcompile_activation/bench.py` |

`zero3_offload_bench.py` at this directory root still runs **model-tensor ZeRO-3** (same flags as before).

## Model-tensor arms

| Arm | `offload_*.pin_memory` | Backend |
|-----|------------------------|---------|
| unpinned | `false` | n/a |
| pinned | `true` | `DS_PIN_MEMORY_BACKEND=native`, `DS_PIN_MEMORY_REGISTER_DEVICE=1` |

`--ablate-register` adds `pinned-unregistered`. CUDA-oriented; skip on XPU if `register_host_memory` is missing (`h2d_d2h/bench.py --skip-native-register`).

```bash
# smoke (synthetic)
python model_tensor_offload/bench.py --hidden 2048 --layers 12 --batch 4 --seq 128

# H200 ZeRO-3 CPU offload (Qwen3 dense + Qwen3.5 hybrid; config + random weights)
# Needs ~256 GiB host RAM for Qwen3-8B without multi-node; raise ulimit -l.
python model_tensor_offload/bench.py --model Qwen/Qwen3-8B --batch 4 --seq 512
python model_tensor_offload/bench.py --model Qwen/Qwen3.5-9B --batch 4 --seq 512

python model_tensor_offload/bench.py --zero-stage 2 --hidden 2048 --layers 12 --batch 4 --seq 128
```

Each arm is a subprocess with a fresh rendezvous port. Results: table plus `DRIVERRESULT=` JSON.
