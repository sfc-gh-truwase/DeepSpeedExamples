# Pin-memory blog: publish tracker

Status of the measurements the blog needs (`blogs/pinned-offload/08-2026/BLOG_PLAN.md`, publish bar).
Every e2e row must be bound to the accelerator's NUMA node (`numactl --cpunodebind=<n> --preferred=<n>`,
as the `run.sh` scripts do), with written host buffers and at least 10 steps. Record `numa_balancing`
and the launch policy for each row.

## Done (H200, bound)

- [x] Model-tensor, Qwen3-8B, seq 4096 (`job-20261024T171244Z`, `20261027T132751Z`)
- [x] Activation checkpoint, Qwen3-8B, seq 8192 and 65536 (`20261027T132751Z`, `20261024T171244Z`)
- [x] Saved-activation, Qwen3-8B, seq 8192 / 16384 / 32768 / 65536 (`20261026T133233Z`, `20261024T171244Z`)
- [x] H2D/D2H microbench, 256 MiB and 1 GiB (`20261027T132751Z`)
- [x] 4K vs 2M page size, all H200 workloads (`20261027T132751Z`)
- [x] Qwen3.5-9B model-tensor (seq 4096) and saved-activation (seq 2K-64K), bound (`20261105T120256Z`, 16K from
  `20261105T162755Z`; Qwen3.5 needs `--discover-calib-seqs 4096 8192` from seq 16384)

## Open

- [ ] **Ascend 910B4, bound re-run.** Model-tensor (Qwen3-4B, seq 1024 and 2048), activation checkpoint
  (Qwen3-4B, seq 8192), saved-activation (Qwen3-4B, seq 2048, now with the H200 flags in `npu/run.sh`:
  `--offload-tensors -1 --credits 1 --prefetch 2`; the unbound 1.14× row used K=3). Current figures are
  unbound. Record `kernel.numa_balancing`, `ulimit -l`, the NUMA node, and the pin backend
  (activation `npu/run.sh` uses `DS_PIN_MEMORY_BACKEND=torch`).
- [ ] **Ascend 910B4, H2D/D2H touched-buffer check.** Re-run native-registered vs pageable at 256 MiB with
  the buffer written and with `--no-touch`, at 4K and 2M. The 6.4 → 23 GB/s page-size claim is unverified.
- [ ] **Arc Pro B60, model-tensor** (Qwen3-0.6B, or 1.7B if host RAM allows). Not collected.
- [ ] **Arc Pro B60, activation checkpoint workload.** Not collected. Open the section with the named engine
  limitation if the engine fails on stock PyTorch XPU.
- [ ] **Arc Pro B60, saved-activation** (Qwen3 0.6B/1.7B, batch 1, seq 2048). Required, not collected.
- [ ] **Arc Pro B60, H2D/D2H microbench** (`--skip-native-register`). Not collected.
- [ ] **H200, Qwen3.5-9B activation checkpoint, bound** at seq 65536 (`MODEL=Qwen/Qwen3.5-9B ./run.sh` in
  `activation_offload/cuda/`). Model-tensor and saved-activation are done above. Current figures are unbound.
- [ ] **Ascend 910B4, Qwen3.5 bound re-run** of model-tensor (seq 512 and 1024) and activation checkpoint
  (seq 8192), plus saved-activation (seq 2048, not yet collected). First confirm `flash-linear-attention` and
  `causal-conv1d` kernels are active: the unbound activation checkpoint step (92.6 s) is ~6× Qwen3-4B's.
  Qwen3.5 on B60 comes with the B60 rows above.
- [x] **Blog README draft** (`blogs/pinned-offload/08-2026/README.md`, 2026-11-05). B60 and bound Ascend
  rows are placeholders until the rows above land.

## Hardware access needed

- B60 and Ascend rows need hardware this tracker's owner does not have in the H200 session. Assign an owner
  per row when picking them up.
- The H200 Qwen3.5 re-run runs on the current GPU pod (`tunji-h200-n1g8-ds-pin-*`) via the autorun queue.
