# H3 sparse Ring experiments and inference

This workflow implements the four-phase H3 placement plan. Offline results and
real runtime measurements are separate artifacts. A CPU process group or a
single GPU does not establish multi-GPU speedup.

## Capture a new real-mask dataset

When the original trace is unavailable, `scripts/analysis/h3_loadbalance/run_h3_experiment.py`
can capture the final selector masks from real two-GPU H3 inference. It runs pure
Ring with all 56 global heads, checks identical mask hashes on both ranks, and
writes one deduplicated rank-zero record per step/layer. These are new captures,
not a reproduction of the unavailable original prompt set. Capture timings are
not performance measurements.

Prepare a JSON list of ten distinct text prompts (indices 0–7 train, 8–9 held out).
The entry point fixes BF16, 256×256, 124 frames, 90% sparsity, 64-token tiles,
exempt mode and five sigma points. Capture requires `--repeats 1`; use fresh
output directories for independent capture runs. It verifies four actual transformer forwards
and records both modalities' actual timesteps and sigmas. A merged VSA checkpoint
can be used directly; this entry point does not attach a separate LoRA.

```bash
python scripts/analysis/h3_loadbalance/run_h3_experiment.py \
  --model /path/to/VSA-trained-H3 --prompts /path/to/prompts.json \
  --output outputs/h3-capture-pilot --conditioning outputs/h3-conditioning \
  --capture --start 0 --count 1 --seed 1000
python scripts/analysis/h3_loadbalance/prepare_trace.py \
  --source outputs/h3-capture-pilot --output outputs/h3-trace-pilot --prompts 1
```

After the pilot validates, use a fresh capture output directory with `--count 10`
and the same conditioning cache, then convert without `--prompts 1`. The converter
rejects missing/duplicate records, mismatched heads, invalid token/tile mappings,
rank disagreement, prefix-mask violations and cross-prompt geometry or schedule
changes. It publishes `video_masks.npy`, `layout_XX.npz`, `provenance.json` and
`source_manifest.json` only after every input validates. The existing planner
still requires exactly ten prompts, four forwards, fifty layers, 56 heads and
forty video tiles.

The capture entry point uses one layerwise-offloaded real text encoder on rank
zero, precomputes the prompt list, and broadcasts each prompt's conditioning.
It explicitly disables DiT layerwise offload so FSDP remains enabled. Its
process-local hooks leave the package implementation unchanged. For generation
comparisons, omit `--capture`, pass `--start 8 --count 1 --repeats 4`, and optionally
add `--layout ... --transport ring` or `demand`. Keep payload logging (`--metrics`)
and profiling (`--profile`) in separate runs. The synchronized denoising-stage
time includes lazy DiT loading; CUDA event spans for each transformer forward
are recorded separately and include device communication waits.

## Phase 1–3: captured-mask experiments

From the repository root, with NumPy, Matplotlib, PyTorch and pytest installed:

```bash
python -m pytest tests/local_tests/h3_loadbalance -q
python scripts/analysis/h3_loadbalance/plan_experiments.py \
  --source /home/kun/output_h3_vsa/predictability-20261007 \
  --output outputs/h3-plan --ranks 8 --cache union \
  --tflops 50 --bandwidth-gbs 25
python scripts/analysis/h3_loadbalance/build_plan_report.py --output outputs/h3-plan
```

Use `--cache separate` in a separate output directory to select policies under
phase-separated caching. The runtime below currently shares fetched KV across
prefix/video queries, so `union` is the closer logical model. Neither model
includes physical Ring forwarding or all collective overheads.

- Phase 1 holds the video 3D layout fixed and sweeps all 16 combinations of
  prefix query spread and audio replica degree for eight ranks. Text KV is
  replicated. Prefix queries are round-robin over ranks `[0, spread)`.
  Per-rank work, send/receive, modality demand, refresh, extra resident memory,
  token destination fan-out and latency proxies are exported. The plotted
  Pareto sets use train means; CSV files retain both train and held-out results.
- Top three policies are selected **only on prompts 0–7**, using the declared
  cache, compute and bandwidth assumptions. The go/no-go compares against packed
  naive. When no-go, later phases are still reported as exploratory diagnostics,
  not as permission to claim a better placement.
- Phase 2 fixes the selected prefix/audio policy, compares contiguous IDs,
  temporal, spatial and 3D partitions, then spectral graph partitioning, greedy
  co-location and bounded swaps. Graph edges are weighted by training
  probability and true KV tile size. It reports selected-edge locality,
  deduplicated video demand, temporal-local/long-range decomposition, spatial
  columns, remote rank fan-out and send/receive imbalance. Video placement is
  again selected on training prompts only. Temporal and contiguous may coincide
  for a trace whose tile IDs are already temporal-major.
- Phase 3 learns a step/layer/head/query prior from the same training prompts.
  Top-2/4/8 predictions report precision, recall, misses, useful/wasted prefetch,
  residual demand and predicted fan-out on held-out prompts. These diagnostics
  isolate the video phase: a dense prefix query can already demand all KV under
  union caching. Prefetch bytes are deduplicated per destination, not per edge.
- Static, per-step, every-ten-layer and prior-drift schedules account for layout
  transfers. Drift means a training-prior mean absolute change above 0.02; it
  is not an oracle using the current held-out mask. Hardware sensitivity uses
  illustrative NVLink, same-NUMA PCIe and cross-NUMA PCIe matrices. A custom
  `--topology-cost costs.json` accepts a nonnegative square seconds/byte matrix
  with zero diagonal. Attention and migration both use these link costs.

Important artifacts:

| Artifact | Meaning |
| --- | --- |
| `phase1.csv`, `phase1_per_rank.csv` | Joint sweep and rank accounting |
| `phase1_selection.json`, `phase1_pareto.png` | Train-only selection and operating points |
| `phase2.csv` | Structured/learned video placement and locality |
| `phase3_prefetch.csv`, `prefetch_top*.npz` | Prediction diagnostics and runtime priors |
| `phase3_adaptation.csv`, `phase3_hardware.csv` | Schedule and topology sensitivity |
| `phase3_boundary.json` | Best evaluated schedule under each assumed topology |
| `runtime_layout.json`, `runtime_*.json` | Deployable fitted ownership policies |
| `plan_provenance.json` | Input/code hashes and experiment assumptions |
| `plan_report.html` | Artifact-derived report |

The fixed source contract is ten prompts, four steps, fifty layers, 56 heads
and forty video tiles. Changing the experiment geometry requires new captured
masks and validation. Exported schedules fail if asked for an uncovered layer
or step; they are not silently repeated. Fitted policies validate the rank count,
video tile count and tile sizes and currently support text/audio plus one video.

## Phase 4: real distributed attention microbenchmark

Two ranks, each on a distinct GPU, with a two-rank fitted policy:

```bash
python scripts/analysis/h3_loadbalance/plan_experiments.py \
  --output outputs/h3-plan-2 --ranks 2 --cache union
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/analysis/h3_loadbalance/benchmark_ring.py \
  --layout outputs/h3-plan-2/runtime_layout.json \
  --prefetch outputs/h3-plan-2/prefetch_top4.npz \
  --output outputs/h3-ring-bench --heads 56 --warmup 3 --repeats 10 --profile
```

This uses captured masks and synthetic QKV, real NCCL/P2P, numerical comparison
against an independent full-attention oracle, measured timing, payload counts,
and optional per-rank PyTorch traces. It compares dense Ring, naive sparse Ring,
static placement, demand transport, imperfect prefetch and adaptive layouts.
It is **not** model generation. Its per-call QKV/output redistribution is
explicitly timed; model inference normally uses persistent hidden ownership.

For CPU correctness testing use `--device cpu` with torchrun/Gloo. The JSON marks
these runs as CPU and never as GPU measurements. A CUDA run rejects more local
processes than available GPUs. The `effective_kv_send_GBs` field divides sent
payload by whole-call elapsed time; it is not an isolated calibrated link rate.

## H3 model inference

Naive VSA Ring on two GPUs:

```bash
python examples/inference/basic/basic_minimax_h3_t2v.py \
  --model-path /path/to/VSA-trained-H3 \
  --attention-backend VIDEO_SPARSE_ATTN_H3 --num-gpus 2 --ring-size 2 \
  --prompt "A red panda walking through a forest" \
  --height 256 --width 256 --num-frames 124 --steps 5 \
  --vsa-tile-size 64 --output outputs/h3-vsa-ring
```

For the published four-forward checkpoint, use
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` as `--model-path`
and add `--vsa-sparsity 0.9`. Its `fastvideo_inference.json` provides the trained
DMD ladder `[999, 749, 500, 250]`; `--steps 5` counts sigma-grid points,
including terminal zero, and therefore runs four transformer forwards.
The current input validator requires at least 124 aligned frames at 24 fps;
the older 22-frame smoke command is rejected.

On discrete GPUs with limited memory, add `--lazy-module-load --no-compile-vae`
to release the text encoder before loading the sharded DiT, then release the
DiT before decoding. This example enables FSDP for multiple GPUs; sequence
sharding alone does not shard the model weights. Start with packed ownership
and no fitted layout: policies from captured masks must match the new request's
geometry. On Ampere GPUs, use the Triton sparse Ring path rather than the
Blackwell `sm100a` kernel. Text-encoder loading and VAE decoding can have their
own memory or dependency limits even when sparse attention itself fits.

Use a VSA-trained checkpoint or an existing inference configuration with a
trained VSA LoRA. The pipeline's guard against untrained, zero compression gates
remains in place. Do not force a dense checkpoint into VSA merely to run this
benchmark. Dense H3 Ring remains available through `FLASH_ATTN`; that main-block
path still requires FlashAttention 2. The new pure sparse Ring path, including
its dense text refiner, uses PyTorch/Triton and does not require FA2.

For a fitted policy, add:

```text
--h3-ring-layout outputs/h3-plan-2/runtime_layout.json
--h3-ring-transport demand
```

The request must match the fitted video geometry. Use a new trace for another
resolution/frame count. `--h3-ring-schedule step`, `every10`, or `drift` chooses
an exported adaptive schedule. `--h3-ring-prefetch .../prefetch_top4.npz` posts
predicted KV transfers before the current selector and fetches missing KV after
selection; unused prefetched tokens never enter the output mask.

The default persistent path migrates hidden states at changed layout boundaries,
keeps residual/FFN computation on the same owner and selects the corresponding
RoPE/AdaLN rows. It restores packed ownership before final normalization and
output projection once per transformer forward. This per-step restoration is
additional to the original offline proxy, which assumed persistent ownership
across step boundaries. `--no-h3-ring-persistent` is a diagnostic fallback that
redistributes QKV and attention output per layer.

Two transport modes are intentionally distinct:

- `ring`: asynchronous neighbor P2P circulates padded canonical KV shards.
  Explicit replica policies do not reduce this forwarding volume.
- `demand`: cyclic **direct** P2P sends the destination's required tokens and
  explicit replicas once, with local attention while the first remote transfer
  is pending. Replica refresh is fused with union demand in the same transfer.
  This is not multi-hop Ring forwarding. Transport sends all heads for each
  selected token; the offline per-head demand model can be smaller.

Both merge partial attention using FP32 online-softmax state. CUDA BF16/FP16
uses a fused Triton kernel and skips key chunks whose mask is entirely false;
CPU/FP32 uses a chunked oracle. Compression-gate outputs use globally reduced
pooled V, and masks use globally reduced pooled Q/K. Split tiles, packed padding,
dense steps/layers, compete mode and reference-video selector spans retain their
existing semantics. Fitted layouts are more narrowly limited as described above.

Sparse Ring is inference-only and currently pure Ring (`ring_size=sp_size`).
Hybrid VSA Ring/Ulysses is rejected. Dense hybrid support is unchanged. Regional
fullgraph compilation remains unsupported for P2P. Fetched/prefetched buffers,
selector scores and workspaces are not covered by the offline explicit-KV-memory
metric; use profiler/device measurements for real peak memory.

## Full generation comparison and observability

```bash
python scripts/analysis/h3_loadbalance/benchmark_inference.py \
  --dense-model /path/to/dense-H3 --vsa-model /path/to/VSA-trained-H3 \
  --num-gpus 2 --layout outputs/h3-plan-2/runtime_layout.json \
  --prefetch outputs/h3-plan-2/prefetch_top4.npz \
  --output outputs/h3-generation --repeats 2 --nsys
```

The harness preserves failed runs, commands, generation times, per-rank logical
payload JSONL and local GPU-utilization samples. Nsight traces are needed to
assess true GPU idle time, NCCL bandwidth and overlap; host wait duration is not
reported as rank idle time. Metrics logging adds overhead and must be kept
consistent across comparisons. For distributed nodes use the existing Ray
launcher with `--execution-backend ray`; telemetry samples only the local node.

Numerical/unit and microbenchmark success does not establish checkpoint-level
video/audio quality or an end-to-end performance improvement. Those conclusions
require successful full-model multi-GPU runs and matched generation settings.
