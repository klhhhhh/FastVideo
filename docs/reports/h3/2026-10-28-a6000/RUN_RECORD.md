# H3 trace 与双卡实验运行记录

报告日期：2026-10-28（America/New_York）。原始程序日志保留主机 UTC 时间。
这是本轮实际运行的记录和复现入口；以下 `outputs/`、`/tmp/` 路径是实验机路径，不要求其他 checkout 已有这些文件。

## 最近六个相关 commit 的功能

| Commit | 已实现功能 | 验证边界 |
| --- | --- | --- |
| `66bf3017` | H3 dense Ring 的 packed sequence padding、局部 partial RoPE 和去 padding | dense Ring 依赖 FA2，不等于 VSA Ring |
| `6a8a8180` | refiner/main block padding、真实分布式 H3 parity 与 Ring 通信测试 | 小随机模型测试不代表真实 checkpoint 质量或加速 |
| `43ced64b` | dense H3 Ring CLI、使用方式和验证限制 | 当时没有真实 VSA mask 采集流程 |
| `11e03a86` | 测试适配更新后的 ring-atten 基础实现 | 测试兼容性调整，没有新增数据采集 |
| `418af92e` | 离线 temporal/spatial/3D、预测布局、通信/迁移计费、单 GPU replay 与报告 | 消费已有 NPY/NPZ；不是真实多 GPU 性能 |
| `089051e8` | pure sparse Ring/Demand、persistent ownership、布局/调度/预取配置、Triton accumulator、两卡 benchmark 与四阶段 planner | 有消费者和推理实现；缺少可复用的真实完整 mask 采集及参数化导出 |

以上按 commit diff 和对应实现核对。本轮代码已提交为 `2eaf1b30`。原实验基于 `089051e8` 加当时尚未提交的脚本；归档保留原 provenance，不将结果冒充为提交后重新运行。

本轮补齐的稳定入口：

- `scripts/analysis/h3_loadbalance/run_h3_experiment.py`：真实 H3 最终 mask hook、跨 rank 哈希验证和去重、单编码器缓存、实际 sigma/forward 记录、A/B/C 运行。
- `scripts/analysis/h3_loadbalance/prepare_trace.py`：packed mask 解码、参数化 shape、严格完整性校验、原子发布 NPY/NPZ 与 manifest。
- `scripts/analysis/h3_loadbalance/summarize_h3.py`：核对三组配置与实际 schedule、统计首次/后续时间、latent/解码输出差异。
- `scripts/analysis/h3_loadbalance/check_pooling_reproducibility.py`：固定合成输入与真实 layout，隔离验证 pooling 数值非确定性。

没有修改核心 attention/model，也没有修复或宣称解决模型输出漂移。

## 实验环境与固定配置

实际路径：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export H3_MODEL=/tmp/h3-a6000-full/model
export H3_TRACE=outputs/h3-new-trace/converted
export H3_PROMPTS=outputs/h3-new-trace/prompts.json
export PYTHONPATH=/tmp/h3-a6000-full/deps
```

`flashinfer-python==0.6.8` 使用独立 overlay，绕过系统 0.7.2 与 CUTLASS 的导入冲突。planner 缺失的 Matplotlib 等依赖另装在 `/tmp/h3-analysis-deps`。版本见 [environment.json](environment.json)。其他机器应准备自己的可用环境，不应照搬临时路径。

原始十条 prompt 已提交为 [prompts.json](prompts.json)。复现时可将 `H3_PROMPTS` 改为该文件路径。固定 seed 1000，256×256、124 帧、BF16、64-token tiles、0.9 sparsity、exempt；五个网格点对应四次真实 forward。当前 runner 固定这些参数，并显式关闭 DiT layerwise offload，保留双卡 FSDP。

## 实际采集与失败记录

1. `outputs/h3-new-trace/pilot/` 首次加载单编码器并生成十条真实 conditioning 缓存；DiT layerwise 默认值意外关闭 FSDP，导致加载 OOM。已显式设置 `dit_layerwise=False`。
2. `outputs/h3-new-trace/pilot-fsdp/` 完成 prompt 0 的 200 条真实 mask 和全部 schedule/latent 记录，但 sample JSON 写入因重复 `prompt_id` 参数失败。修复后从同次保存的 measurement/config 恢复 metadata；恢复说明写入 sample，转换器独立逐条验证全部 200 条后才接受。
3. `pilot-validated/` 单 prompt 转换通过，随后将已验收 prompt 0 复制进 `raw/`，继续采集 prompt 1–9。没有补零或随机造 mask。

实际成功采集命令：

```bash
python scripts/analysis/h3_loadbalance/run_h3_experiment.py \
  --model "$H3_MODEL" --prompts "$H3_PROMPTS" \
  --output outputs/h3-new-trace/raw \
  --conditioning outputs/h3-new-trace/conditioning \
  --capture --start 1 --count 9 --seed 1000
python scripts/analysis/h3_loadbalance/prepare_trace.py \
  --source outputs/h3-new-trace/raw --output "$H3_TRACE"
```

完整新机器复现时，应先用一个新的 output 运行 `--capture --start 0 --count 1` 并以 `prepare_trace.py --prompts 1` 验收；再选择另一个新 output 运行 `--capture --start 0 --count 10`。不要在已有目录混写，也不要用 `--capture --repeats 4`：稳定入口现在拒绝多次采集覆盖同名记录。

hook 位于 sparse Ring 调用的最终 `_build_block_mask` 返回位置，保存完整 prefix/video mask；每条检查两 rank 的全局 mask SHA256 相等，写一份全局 56-head 数据。layout 包含 tile 实际大小、模态、packed token→tile、position IDs。最终导出布尔 `(10,4,50,56,40,40)`；其他 shape 可以用 converter 的 `--steps/--layers/--heads/--video-tiles/--selected-tiles` 显式声明，但现有 planner 仍只接受原固定几何。

## Planner 与真实 NCCL correctness gate

```bash
PYTHONPATH=/tmp/h3-analysis-deps python scripts/analysis/h3_loadbalance/plan_experiments.py \
  --source "$H3_TRACE" --output outputs/h3-plan-2 --ranks 2 --cache union
PYTHONPATH=/tmp/h3-analysis-deps python scripts/analysis/h3_loadbalance/build_plan_report.py \
  --output outputs/h3-plan-2
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/analysis/h3_loadbalance/benchmark_ring.py \
  --source "$H3_TRACE" --layout outputs/h3-plan-2/runtime_layout.json \
  --device cuda --heads 2 --warmup 1 --repeats 2 \
  --output outputs/h3-ring-smoke-2gpu
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/analysis/h3_loadbalance/benchmark_ring.py \
  --source "$H3_TRACE" --layout outputs/h3-plan-2/runtime_layout.json \
  --device cuda --heads 56 --prompt 8 --step 2 --layer 20 \
  --warmup 3 --repeats 10 --output outputs/h3-ring-bench-2gpu
```

planner 首次在绘图时缺 Matplotlib，补齐隔离依赖后重跑；旧失败日志仍在 `outputs/h3-new-trace/plan-missing-matplotlib.log`。
56-head 另依次运行 `(9,2,20)`、`(8,0,0)`、`(9,3,49)`，各自写入 `outputs/h3-ring-bench-p9-s2-l20`、`h3-ring-bench-p8-s0-l0`、`h3-ring-bench-p9-s3-l49`。
未传 `--prefetch`。benchmark 自带的调度路径也执行了 oracle 检查；完整模型比较只使用 fixed 静态布局。

## A/B/C 四次性能运行

三个进程组顺序执行，避免同时占 GPU。入口自己启动两 worker，不再套 torchrun。每组均复用真实文本缓存，性能运行不打开 metrics/profile：

```bash
for case in A B C; do
  extra=()
  transport=ring
  if [ "$case" != A ]; then
    extra+=(--layout outputs/h3-plan-2/runtime_layout.json)
  fi
  if [ "$case" = C ]; then
    transport=demand
  fi
  python scripts/analysis/h3_loadbalance/run_h3_experiment.py \
    --model "$H3_MODEL" --prompts "$H3_PROMPTS" \
    --conditioning outputs/h3-new-trace/conditioning \
    --output "outputs/h3-abc-seed1000/$case" \
    --start 8 --count 1 --seed 1000 --repeats 4 \
    --transport "$transport" "${extra[@]}"
done
python scripts/analysis/h3_loadbalance/summarize_h3.py --root outputs/h3-abc-seed1000
```

这是与保存 argv 等价的 shell 展开；精确历史 argv、exit code、每次输出均在各组本机目录。已经存在的结果目录不可直接重复使用。首次单列，后续三次取中位/范围/总体标准差；allocator 高水位未逐次 reset。CUDA event forward span 包含设备通信等待，stage 时间另含加载。

## 独立观测、profiler 与一致性诊断

每个 case 另起一次进程，用相同参数改为 `--repeats 1` 并换新 output：

- payload：加 `--metrics <新输出目录>/metrics`，三组都有独立文件。
- profiler：加 `--profile`，不加 metrics；导出 `profile_rank_0.json` / `profile_rank_1.json`。本机没有 nsys，使用 PyTorch CPU/CUDA trace；累计 kernel duration 不是 GPU idle 或独立带宽。

当时额外的 fingerprint runner 已保存在 [repeat_diagnostic.py.snapshot](repeat_diagnostic.py.snapshot)，内容为实际运行的历史源代码，不是当前稳定入口。可复制到一个临时 `.py` 文件，再传同样 CLI 运行。它在采集时额外记录初始 noise/conditioning、首层 Q/K 与 score 哈希；A 用 `--capture --repeats 2`，按 repeat 分目录，B/C 各一次。该多次采集能力仅存在于此历史诊断快照；不要用于性能计时。历史输出为 `A-observation/`、`B-observation/`、`C-observation/`，均开启独立 metrics。

pooling 隔离控制的稳定复现命令：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/analysis/h3_loadbalance/check_pooling_reproducibility.py \
  --layout "$H3_TRACE/layout_08.npz" \
  --output outputs/h3-pooling-control/pooling.json --repeats 12
```

它只对固定合成输入重复实际 `pooled_tiles`，并对比 deterministic algorithms 关闭/开启；不验证确定性全模型，也不产生模型加速结论。

## 本轮提交前复核

```bash
PYTHONPATH=/tmp/h3-a6000-full/deps python -m pytest tests/local_tests/h3_loadbalance -q
pre-commit run --files <本轮代码和文档路径>
git diff --check
```

完整套件的结果、报告中保留的失败结论与 [REPORT.md](REPORT.md) 一致。报告目录只收入精简证据，原始视频/tensor/profiler trace 继续留在 ignored outputs，不强行提交大文件。
