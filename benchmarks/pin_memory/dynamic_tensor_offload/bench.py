# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Dynamic ZeRO-3 state offload (DeepSpeedEngine.offload_states/reload_states,
DeepSpeed PR #6011): pinned vs unpinned host memory.

Unlike model_tensor_offload (static offload_param/offload_optimizer baked
into the ZeRO config), this benchmark keeps the model fully GPU-resident at
construction. CPU residency is produced only by explicit
engine.offload_states()/reload_states() calls each step -- and only for the
state that is actually idle between steps:

    reload_states()  -- bring hp_params (fp32 master weights) and Adam
                         state back to the device
    forward/backward/step  -- fully GPU-resident compute
    offload_states() -- move hp_params + Adam state back to host (pinned
                         or pageable)

hp_params and optim_states are only touched by the optimizer step, so they
are the only two OffloadStateTypeEnum members this benchmark round-trips.
lp_params (bf16 weights used by forward/backward), lp_grads and
contiguous_grad_buffer stay GPU-resident the whole run: lp_grads/
contiguous_grad_buffer are produced fresh by backward every step (nothing to
"reload" -- reload_states() on contiguous_grad_buffer is just an empty
re-allocation, no host transfer), and lp_params must stay live because
DeepSpeedZeroOptimizer_Stage3.step() downcasts the just-updated hp_params
into it in place -- offloading lp_params between steps would make that
write-back a no-op/stale read. reload_states() has no `include` filter of
its own; it reloads whatever subset is currently present in
self.offloaded_states, so narrowing what offload_states(include=...) adds to
that set is what makes reload selective here, not an argument to
reload_states() itself. (Confirmed by reading
deepspeed/runtime/zero/stage3.py's offload_states/reload_states directly on
the fork.)

offload_states() asserts the ZeRO config has no static offload_param /
offload_optimizer (DeepSpeedEngine.offload_states: "Moving states across
devices is not supported for offloaded optimizer/parameters"), so --zero-stage
is fixed at 3 with neither block set. It also hard-asserts the optimizer is
DeepSpeed's FusedAdam (deepspeed/runtime/zero/stage3.py:offload_states), which
is what plain {"type": "AdamW"} with no CPU-offload optimizer config resolves
to in DeepSpeedEngine._configure_basic_optimizer.

Note on what "pin" measures here: offload_states()/reload_states() pin their
host buffers via accelerator.pin_memory(), the same call model_tensor_offload
uses for its own offload_param/offload_optimizer pinning. That call checks
DS_PIN_MEMORY_BACKEND and, when it is "native", routes through this fork's
NativePinnedMemory pool (deepspeed/utils/pin_memory.py) instead of a plain
torch cudaHostAlloc. Like model_tensor_offload/bench.py, this script sets
DS_PIN_MEMORY_BACKEND=native up front so the "pinned" arm actually exercises
the fork's pool, not just torch's pinning. Buffers are allocated once (first
offload_states() call) and reused on every subsequent call, so --warmup must
be >= 1 for the timed steps to reflect steady-state transfer cost rather than
one-time pin-buffer allocation.

Multi-GPU / real ZeRO-3 sharding: unlike every other benchmark in this suite
(model_tensor_offload, activation_offload, saved_activation_offload), which
run through common.py's dist_env() -- a single-process WORLD_SIZE=1 stub, so
their ZeRO-3 config never actually shards anything -- this benchmark's driver
launches each arm via `torchrun --standalone --nproc-per-node=<world-size>`
(default 8) so ZeRO-3 genuinely partitions params/optimizer state across
real ranks/GPUs. Only rank 0 measures and prints ARMRESULT; its GPU peak and
offload_gb numbers are what one real GPU actually holds under this
world_size, not an unsharded single-process total. run_arm() detects a
torchrun launch (TORCHELASTIC_RUN_ID in the environment) and skips
common.py's dist_env() in that case -- calling dist_env() under torchrun
would stomp every rank's real RANK/WORLD_SIZE back to 0/1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import (build_hf_causal_lm, dist_env, param_numel, parse_arm_result, patch_cpp_extension_drop_cxx17,
                    print_arm_result, print_driver_result)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="HF model id; architecture from config, random weights. Omit for synthetic.")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4, help="micro batch size PER GPU")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--zero-stage",
                        type=int,
                        choices=(3, ),
                        default=3,
                        help="offload_states/reload_states require ZeRO-3.")
    parser.add_argument("--world-size", type=int, default=8, help="driver only: GPUs per arm via torchrun")
    parser.add_argument("--pin", type=int, default=None, help="internal: offload_states pin_memory 0/1")
    return parser.parse_args()


def _build_synthetic(hidden, layers):
    import torch

    class Block(torch.nn.Module):

        def __init__(self, h):
            super().__init__()
            self.fc1 = torch.nn.Linear(h, 4 * h)
            self.fc2 = torch.nn.Linear(4 * h, h)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Net(torch.nn.Module):

        def __init__(self, h, n):
            super().__init__()
            self.emb = torch.nn.Embedding(32000, h)
            self.blocks = torch.nn.ModuleList([Block(h) for _ in range(n)])
            self.head = torch.nn.Linear(h, 32000, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            for block in self.blocks:
                x = x + block(x)
            return self.head(x).sum()

    return Net(hidden, layers), 32000


def _cpu_bytes(tensor):
    import torch
    if tensor is None or not torch.is_tensor(tensor):
        return 0
    if tensor.device.type != "cpu":
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _measure_offload_gb(engine):
    """CPU-resident param shards + optimizer/master-weight tensors right after offload_states()."""
    seen = set()
    param_bytes = 0
    optim_bytes = 0

    def add(bucket, tensor):
        nonlocal param_bytes, optim_bytes
        nbytes = _cpu_bytes(tensor)
        if nbytes == 0:
            return
        key = int(tensor.data_ptr())
        if key in seen:
            return
        seen.add(key)
        if bucket == "param":
            param_bytes += nbytes
        else:
            optim_bytes += nbytes

    for param in engine.module.parameters():
        add("param", getattr(param, "ds_tensor", param))
        add("param", param)

    optimizer = engine.optimizer
    for tensor in getattr(optimizer, "fp16_partitioned_groups_flat", []) or []:
        add("param", tensor)
    for tensor in getattr(optimizer, "fp32_partitioned_groups_flat", []) or []:
        add("optim", tensor)
        add("optim", getattr(tensor, "grad", None))
        add("optim", getattr(tensor, "exp_avg", None))
        add("optim", getattr(tensor, "exp_avg_sq", None))
        add("optim", getattr(tensor, "exp_avg_cpu_data", None))
        add("optim", getattr(tensor, "exp_avg_sq_cpu_data", None))

    inner = getattr(optimizer, "optimizer", optimizer)
    state = getattr(inner, "state", None)
    if isinstance(state, dict):
        for group_state in state.values():
            if not isinstance(group_state, dict):
                continue
            for value in group_state.values():
                add("optim", value)

    gb = 1024**3
    return {
        "offload_param_gb": round(param_bytes / gb, 2),
        "offload_optim_gb": round(optim_bytes / gb, 2),
        "offload_gb": round((param_bytes + optim_bytes) / gb, 2),
    }


def _zero_config(stage):
    # No offload_param / offload_optimizer: offload_states() asserts neither
    # is configured. State lives on GPU except during the explicit
    # offload_states()..reload_states() window each step.
    return {"stage": stage}


def _under_torchrun():
    return "TORCHELASTIC_RUN_ID" in os.environ


def run_arm(args):
    # Mirrors model_tensor_offload/bench.py: route accelerator.pin_memory()
    # through the fork's native pinned-memory pool instead of plain torch
    # pinning. Only matters when offload_states(pin_memory=True) below
    # actually asks to pin -- the unpinned arm never calls pin_memory().
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "1" if args.pin else "0"
    if _under_torchrun():
        # torchrun already set RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/
        # MASTER_PORT correctly for real multi-rank ZeRO-3 sharding --
        # dist_env() would stomp them back to a fake single-process 0/1.
        pass
    else:
        dist_env()
    patch_cpp_extension_drop_cxx17()

    import torch
    import deepspeed
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    # Only hp_params (fp32 master weights) and optim_states (Adam exp_avg/
    # exp_avg_sq) are idle between steps -- stage just those two. lp_params
    # stays GPU-resident (needed by forward/backward and by step()'s
    # downcast write-back); lp_grads/contiguous_grad_buffer stay resident
    # too (freshly produced by backward every step, nothing to round-trip).
    offload_include = [OffloadStateTypeEnum.hp_params, OffloadStateTypeEnum.optim_states]

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_optimization": _zero_config(args.zero_stage),
    }

    hf_transformers = None
    hf_model_type = None
    if args.model:
        # remote_device left at its default (GPU): nothing is statically
        # offloaded here, unlike model_tensor_offload's remote_device="cpu".
        # Under real world_size>1, zero.Init() shards initialization across
        # ranks instead of each rank materializing the full model.
        with deepspeed.zero.Init(config_dict_or_path=ds_config, dtype=torch.bfloat16):
            model, vocab, hf_transformers, hf_model_type = build_hf_causal_lm(args.model, torch.bfloat16)

        def forward_loss(engine, ids):
            return engine(input_ids=ids, labels=ids).loss
    else:
        model, vocab = _build_synthetic(args.hidden, args.layers)

        def forward_loss(engine, ids):
            return engine(ids)

    n_params = sum(param_numel(p) for p in model.parameters())

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    rank = engine.global_rank
    world_size = engine.world_size
    dev = engine.device
    dev_api = getattr(torch, dev.type)
    ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=dev)

    reload_times = []
    compute_times = []
    offload_times = []
    for step in range(args.warmup + args.steps):
        if step > 0:
            dev_api.synchronize()
            t0 = time.perf_counter()
            engine.reload_states(non_blocking=False)
            dev_api.synchronize()
            reload_s = time.perf_counter() - t0
        else:
            reload_s = 0.0

        dev_api.synchronize()
        t0 = time.perf_counter()
        loss = forward_loss(engine, ids)
        engine.backward(loss)
        engine.step()
        dev_api.synchronize()
        compute_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        engine.offload_states(include=offload_include,
                              device=OffloadDeviceEnum.cpu,
                              pin_memory=bool(args.pin),
                              non_blocking=False)
        dev_api.synchronize()
        offload_s = time.perf_counter() - t0

        if step >= args.warmup:
            reload_times.append(reload_s)
            compute_times.append(compute_s)
            offload_times.append(offload_s)

    if rank != 0:
        return

    offload = _measure_offload_gb(engine)

    def _mem(fn):
        try:
            return round(fn() / 1e9, 2)
        except Exception:
            return None

    transfer_times = [r + o for r, o in zip(reload_times, offload_times)]
    print_arm_result({
        "experiment": "dynamic_tensor_offload",
        "zero_stage": args.zero_stage,
        "world_size": world_size,
        "pin_memory": bool(args.pin),
        "device": dev.type,
        "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
        "hf_model_type": hf_model_type,
        "transformers": hf_transformers,
        "params_b": round(n_params / 1e9, 3),
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": args.batch * args.seq * world_size,
        "steps": len(transfer_times),
        "reload_avg_s": sum(reload_times) / len(reload_times),
        "compute_avg_s": sum(compute_times) / len(compute_times),
        "offload_avg_s": sum(offload_times) / len(offload_times),
        "transfer_avg_s": sum(transfer_times) / len(transfer_times),
        "transfer_min_s": min(transfer_times),
        "gpu_peak_gb": _mem(dev_api.max_memory_allocated),
        **offload,
    })


def _passthrough(args):
    extra = [
        "--zero-stage",
        str(args.zero_stage),
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
    ]
    if args.model:
        extra.extend(["--model", args.model])
    return extra


def run_arm_torchrun(script_path, extra_args, world_size):
    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={world_size}",
        os.path.abspath(script_path),
        *extra_args,
    ]
    print(f"[driver] {' '.join(command)}", flush=True)
    proc = subprocess.run(command, env=os.environ.copy(), capture_output=True, text=True)
    arm = parse_arm_result(proc.stdout)
    if arm is None:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"arm produced no ARMRESULT (rc={proc.returncode})")
    return arm


def run_driver(args):
    script = os.path.abspath(__file__)
    results = {}
    for name, pin in (("unpinned", 0), ("pinned", 1)):
        extra = _passthrough(args) + ["--pin", str(pin)]
        results[name] = run_arm_torchrun(script, extra, args.world_size)

    unpinned = results["unpinned"]
    pinned = results["pinned"]
    print("\n================ Dynamic ZeRO-3 state offload (offload_states/reload_states) ================")
    print(f"stage: {pinned['zero_stage']}  world_size: {pinned['world_size']}  model: {pinned['model']}  "
          f"params: {pinned['params_b']}B (full, unsharded)  device: {pinned['device']}")
    print(f"batch/gpu: {pinned['batch']} x seq: {pinned['seq']}  "
          f"({pinned['tokens_per_step']} tokens/step across all GPUs)")
    print(f"offload PER GPU (hp_params + optim_states only, sharded 1/{pinned['world_size']}): "
          f"params {pinned.get('offload_param_gb', 0):.2f} GB  "
          f"optimizer {pinned.get('offload_optim_gb', 0):.2f} GB  "
          f"total {pinned.get('offload_gb', 0):.2f} GB")
    print()
    print(f"{'arm':<10}{'reload (s)':>12}{'offload (s)':>13}{'transfer (s)':>14}{'compute (s)':>13}"
          f"{'GPU peak (GB)':>16}")
    for name in ("unpinned", "pinned"):
        row = results[name]
        print(f"{name:<10}{row['reload_avg_s']:>12.3f}{row['offload_avg_s']:>13.3f}"
              f"{row['transfer_avg_s']:>14.3f}{row['compute_avg_s']:>13.3f}"
              f"{(row['gpu_peak_gb'] or 0):>16.2f}")
    speedup = unpinned["transfer_avg_s"] / pinned["transfer_avg_s"]
    saved = unpinned["transfer_avg_s"] - pinned["transfer_avg_s"]
    compute_delta_pct = 100.0 * (pinned["compute_avg_s"] - unpinned["compute_avg_s"]) / unpinned["compute_avg_s"]
    print()
    print(f"pinning speedup (reload+offload transfer only): {speedup:.2f}x   saved: {saved:.3f} s/step")
    print(f"compute step time is offload-invariant by construction: pinned vs unpinned differ by "
          f"{compute_delta_pct:+.1f}% (noise, not a pin effect)")
    print_driver_result({"unpinned": unpinned, "pinned": pinned, "speedup": speedup})


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
