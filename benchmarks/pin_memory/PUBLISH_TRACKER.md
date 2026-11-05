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

## Open

- [ ] **Ascend 910B4, bound re-run.** Model-tensor (Qwen3-4B, seq 1024 and 2048), activation checkpoint
  (Qwen3-4B, seq 8192), saved-activation (Qwen3-4B, seq 2048). Current figures are unbound.
- [ ] **Ascend 910B4, H2D/D2H touched-buffer check.** Re-run native-registered vs pageable at 256 MiB with
  the buffer written and with `--no-touch`, at 4K and 2M. The 6.4 → 23 GB/s page-size claim is unverified.
- [ ] **Arc Pro B60, model-tensor** (Qwen3-0.6B, or 1.7B if host RAM allows). Not collected.
- [ ] **Arc Pro B60, activation checkpoint workload.** Not collected. Open the section with the named engine
  limitation if the engine fails on stock PyTorch XPU.
- [ ] **Arc Pro B60, saved-activation** (Qwen3 0.6B/1.7B, batch 1, seq 2048). Required, not collected.
- [ ] **Arc Pro B60, H2D/D2H microbench** (`--skip-native-register`). Not collected.
- [ ] **H200, Qwen3.5-9B bound re-run** of model-tensor and saved-activation (current figures unbound).
- [ ] **Ascend 910B4, Qwen3.5 bound re-run** of model-tensor (current figures unbound). Qwen3.5 on B60 comes
  with the B60 rows above.
- [ ] **Blog README draft** (`blogs/pinned-offload/08-2026/README.md`) after the rows above are in.

## Hardware access needed

- B60 and Ascend rows need hardware this tracker's owner does not have in the H200 session. Assign an owner
  per row when picking them up.
- The H200 Qwen3.5 re-run runs on the current GPU pod (`tunji-h200-n1g8-ds-pin-*`) via the autorun queue.
