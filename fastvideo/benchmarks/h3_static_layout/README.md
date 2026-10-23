# FastH3 Static Layout Experiments

This package implements CPU-only simulators for the FastH3 multimodal static
layout experiments. The goal is to evaluate prefix query spread and audio KV
residency policies before wiring anything into the real H3 runtime.

These scripts do **not** load MiniMax-H3 weights, call `VideoGenerator`, run
Ring Attention, or execute VSA kernels. They model token ownership, estimated
attention work, and estimated communication for a distributed layout.

## Purpose

The experiments answer one narrow question:

> For a fixed H3-style multimodal sequence, how should dense text/audio prefix
> queries be spread across ranks, and how much audio KV should be replicated,
> before communication costs erase the load-balance gain?

The simulator produces the metrics needed to compare operating points:

- load-balance ratio `rho = max(rank_work) / mean(rank_work)`
- per-rank token-pair work
- remote KV demand bytes
- text/audio/video remote bytes
- audio replica refresh bytes
- extra audio KV memory
- max send and receive bytes per rank
- destination fan-out
- latency proxy
- Pareto frontiers for communication vs rho and communication vs latency

## Scripts

### Prefix Query Spread

Sweeps `prefix_spread`, while keeping `audio_replicas` fixed.

```bash
python -m fastvideo.benchmarks.h3_static_layout.prefix_query_spread \
  --world-size 8 \
  --prefix-spreads 1,2,4,8 \
  --audio-replicas 1 \
  --output-dir bench/h3_static_layout/prefix_query_spread
```

Use this to check the expected trend:

```text
prefix_spread increases -> rho decreases, communication increases
```

### Audio KV Residency

Sweeps `audio_replicas`, while keeping `prefix_spread` fixed.

```bash
python -m fastvideo.benchmarks.h3_static_layout.audio_kv_residency \
  --world-size 8 \
  --audio-replicas 1,2,4,8 \
  --prefix-spread 1 \
  --output-dir bench/h3_static_layout/audio_kv_residency
```

Use this to compare:

```text
saved remote audio demand - audio replica refresh cost
```

### Joint Static Layout Sweep

Runs the full `prefix_spread x audio_replicas` grid.

```bash
python -m fastvideo.benchmarks.h3_static_layout.joint_sweep \
  --world-size 8 \
  --prefix-spreads 1,2,4,8 \
  --audio-replicas 1,2,4,8 \
  --output-dir bench/h3_static_layout/joint_sweep
```

This is the main experiment. It emits the two Pareto views used to choose
candidate static layouts:

- communication bytes vs `rho`
- communication bytes vs latency proxy

## Output Files

Each script writes three artifacts:

```text
<experiment>.json
<experiment>.csv
<experiment>.md
```

The JSON contains complete structured results, including Pareto frontiers and
recommended layouts. The CSV contains scalar metrics for quick plotting. The
Markdown report is a compact human-readable summary.

## Implementation

`model.py` contains the shared simulator:

- `H3StaticLayoutShape` describes the token counts and cost assumptions.
- `H3StaticLayoutPolicy` describes one static layout.
- `simulate_static_layout_policy()` computes metrics for one policy.
- `run_static_layout_grid()` evaluates a policy grid.
- `pareto_frontier()` and `recommend_static_layouts()` rank layouts.

The model assumes:

- text KV is fully replicated because it is small;
- audio KV is replicated in contiguous replica groups;
- video Q is evenly split across ranks with a fixed local-window sparse policy;
- prefix Q attends to all text, audio, and video KV;
- video Q attends to all text/audio KV plus a local selected video window;
- communication is estimated from remote KV ownership, not real NCCL traces.

`reporting.py` writes JSON, CSV, and Markdown artifacts. The experiment modules
only define the sweep axes and CLI defaults.

## Important Limits

This is not a real H3 inference benchmark. It does not prove wall-clock speedup.
It is a planning tool for picking candidate layouts to implement later in the
distributed attention runtime.

`world_size=1` is intentionally rejected. A single-rank run has no remote KV
movement and cannot evaluate prefix spread or audio replication trade-offs.

To validate runtime speedup, a separate implementation must connect the chosen
layout to H3 attention execution and compare against a real Ring/SP baseline.
