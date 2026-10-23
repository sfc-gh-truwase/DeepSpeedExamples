# Dynamic ZeRO-3 state offload

Pin vs pageable transfer time for DeepSpeed's *dynamic* state offload API
(`DeepSpeedEngine.offload_states()` / `.reload_states()`, DeepSpeed PR #6011),
as opposed to [`model_tensor_offload`](../model_tensor_offload/README.md)'s
*static* `offload_param`/`offload_optimizer` ZeRO config.

The ZeRO config here has neither `offload_param` nor `offload_optimizer`:
`offload_states()` asserts both are unset, so the model is fully GPU-resident
at construction. The optimizer must be DeepSpeed's `FusedAdam` (what plain
`{"type": "AdamW"}` with no CPU-offload optimizer config resolves to) --
`offload_states()` hard-asserts this.

Only `--zero-stage 3` is supported.

**Only hp_params + optim_states are round-tripped.** `reload_states()` has no
selective/`include` argument of its own -- it unconditionally reloads
whatever subset of `OffloadStateTypeEnum` is currently present in
`self.offloaded_states` (confirmed by reading
`deepspeed/runtime/zero/stage3.py` directly), so narrowing what
`offload_states(include=...)` adds to that set is what makes reload
selective, not an argument to `reload_states()`. This benchmark's step loop
only offloads `hp_params` (fp32 master weights) and `optim_states` (Adam
`exp_avg`/`exp_avg_sq`) -- the only state actually idle between steps.
`lp_params` (bf16 weights used by forward/backward) stays GPU-resident the
whole run, because `DeepSpeedZeroOptimizer_Stage3.step()` downcasts the
just-updated `hp_params` into it in place; offloading it between steps would
make that write-back a no-op/stale read. `lp_grads`/`contiguous_grad_buffer`
also stay resident -- they're produced fresh by backward every step, so
there's nothing to round-trip (`reload_states()`'s "restore" of
`contiguous_grad_buffer` is just an empty re-allocation with no host
transfer, by its own source comment).

**Real multi-GPU ZeRO-3 sharding -- unlike this suite's other benchmarks.**
`model_tensor_offload`, `activation_offload` and `saved_activation_offload`
all run through `common.py`'s shared `dist_env()`, which hardcodes
`WORLD_SIZE=1` -- so their ZeRO-3 config never actually shards anything;
their "offload_gb"/capacity numbers are the full, *unsharded* footprint for
that model on one process. This benchmark's driver instead launches each arm
via `torchrun --standalone --nproc-per-node=<world-size>` (`--world-size`,
default 8), so ZeRO-3 genuinely partitions params/optimizer state across
real ranks/GPUs -- `run_arm()` detects the torchrun launch
(`TORCHELASTIC_RUN_ID` in the environment) and skips `dist_env()`'s
single-process override in that case. Only rank 0 measures and prints
`ARMRESULT`; every rank still runs the full
reload/forward/backward/step/offload loop every iteration (these involve
collective ops), but only rank 0's local GPU-memory/offload numbers are
reported -- i.e. they reflect what *one* real GPU holds under this
world_size, not an unsharded single-process total.

**Earlier single-process (world_size=1) finding, now superseded.** Before
this harness launched real multi-rank, the single-process config (same
config as every other benchmark in this suite) measured Qwen3.5-9B's full
unsharded footprint at ~150 GB -- that OOMed on a single H200 even with
hp_params/optim_states-only staging (135.85 GiB in use, failed allocating
6.57 GiB more against 139.80 GiB capacity, inside `FusedAdam.step()`
creating `exp_avg_sq`), so a downsized Qwen3-4B was used as the matched
workload instead. That was never a real "9B needs 150GB per GPU under
ZeRO-3" result -- it was what one *unsharded* process needs, which is what a
single-process harness necessarily measures. At real `world_size=8` (below),
Qwen3.5-9B's true per-GPU footprint is ~1/8 of that and fits with large
headroom.

**"pin" exercises this fork's native pinned-memory pool.**
`accelerator.pin_memory()` checks `DS_PIN_MEMORY_BACKEND` and, when it's
`"native"`, routes through this fork's `NativePinnedMemory` pool
(`deepspeed/utils/pin_memory.py`) instead of a plain torch `cudaHostAlloc` --
confirmed by reading `CUDA_Accelerator.pin_memory` directly. Like
`model_tensor_offload/bench.py`, this script sets `DS_PIN_MEMORY_BACKEND=native`
up front, so the pinned arm here exercises the same pool
`model_tensor_offload`/`saved_activation_offload` do, not independent torch
pinning. Buffers are allocated once, on the first `offload_states()` call,
and reused on every later call, so `--warmup` must stay >= 1 for the
reported numbers to reflect steady-state transfer cost rather than one-time
pin allocation.

Reports `reload (s)` / `offload (s)` / `transfer (s)` (= reload + offload,
the pin-vs-pageable comparison metric) and `compute (s)` (forward/backward/
step on fully GPU-resident state -- expected to be flat across arms, since
compute never touches the offloaded buffers; reported as a sanity check --
in practice we've seen it vary run-to-run more than that logic suggests, so
treat large compute deltas between arms as noise, not a pin effect, unless
reproduced).

See the [parent README](../README.md).

## Results: H200 x8, Qwen3.5-9B, ZeRO-3, world_size=8 (real sharding)

```bash
ulimit -l unlimited
python bench.py --model Qwen/Qwen3.5-9B --batch 4 --seq 512 --world-size 8
```

Driver launches each arm as `torchrun --standalone --nproc-per-node=8
bench.py ... --pin {0,1}`; only rank 0's numbers are reported below.

Measured (batch 4/GPU x seq 512, 8.954B params full/unsharded, offload
**12.51 GB/GPU** -- hp_params + optim_states only, truly sharded 1/8; GPU
peak **46.30 GB/GPU**, comfortably under one H200's 140GB):

| arm      | reload (s) | offload (s) | transfer (s) | compute (s) |
|----------|-----------:|-------------:|-------------:|------------:|
| unpinned |      2.278 |        8.564 |       10.842 |       4.181 |
| pinned   |      1.896 |        3.133 |        5.029 |       4.839 |

Pinning speedup (reload+offload transfer only): **2.16x** (saved 5.81
s/step). Compute step time differs by +15.7% between arms (noise, not a pin
effect -- by construction compute never touches the offloaded buffers).

## Historical: H200 x1, Qwen3-4B, ZeRO-3, world_size=1 (unsharded, pre-fix)

Kept for reference only -- this predates the multi-GPU launch above and used
the same single-process harness as every other benchmark in this suite
(hence the Qwen3-4B downsize, see the superseded finding above).

```bash
ulimit -l unlimited
python bench.py --model Qwen/Qwen3-4B --batch 4 --seq 512
```

Measured (batch 4 x seq 512, 4.411B params, offload 49.30 GB -- hp_params +
optim_states only, lp_params stays resident so offload_param_gb reads 0.00 --
GPU peak 87.98 GB both arms):

| arm      | reload (s) | offload (s) | transfer (s) | compute (s) |
|----------|-----------:|-------------:|-------------:|------------:|
| unpinned |      8.517 |       13.279 |       21.796 |       1.163 |
| pinned   |      1.378 |        1.937 |        3.315 |       0.824 |

Pinning speedup (reload+offload transfer only): 6.57x (saved 18.5 s/step).
