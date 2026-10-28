# H3 双卡静态布局实验结果

这轮计划的采集、两卡 oracle 校验、A/B/C 各四次生成、独立 payload 观测和独立 CPU/CUDA profiling 均已执行。
**本次未观察到静态布局加速；完整模型的严格输出一致性检查未通过。** Attention 的固定-mask oracle 校验通过，不能据此宣称 checkpoint 级输出严格等价。

## 数据与运行配置

- 硬件：2 × RTX A6000，跨 NUMA PCIe，无 NVLink；实际采用 FSDP 分片 DiT。
- 模型：`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` 官方合并导出，step 1300；未另挂 LoRA。完整模型声明/provenance 已嵌入各组 `run_config.json`。
- 这是重新采集的数据：十条新 prompt，0–7 拟合布局，8–9 held-out；不是缺失的原始 trace 的逐样本复现。
- BF16，256×256，124 帧、24 fps，seed 1000，VSA sparsity 0.9、64-token tile、exempt；无 dense override。
- 实际四次 transformer forward：DMD rungs `[999,749,500,250]`，五个 sigma 网格点。两种模态实际 sigma/timestep 均保存并核对相同。
- 两卡采集完整最终 mask，逐条验证两 rank 哈希一致，只保留 rank 0 的全局 56-head 记录，未拼接 rank 文件。
- 共 2,000 条 step/layer 记录；导出 `video_masks.npy` 为 `(10,4,50,56,40,40)`，布尔类型。每个 video query 保留四个 video KV tiles。
- 全部记录、tile/token 映射、prefix dense 规则、跨 prompt 几何和 schedule 通过验收。无缺记录补零。
- 性能请求固定 held-out prompt 8（乡间道路骑车人），fixed + persistent，无预取，无 Transformer/VAE compilation。A 不传 layout；B/C 使用同一份训练侧选出的 layout，分别为 Ring/Demand。
- 文本 conditioning 使用真实编码缓存。各阶段惰性加载/释放，因此每次重复包含 DiT/decoder 重载成本。最终 latent 读回保存有小量额外开销，各组相同。

## 完整模型性能

每组四次，首次单列；后续统计仅三个样本。生成时间包括组件重载与解码；CUDA span 是每次四个 transformer forward 的事件跨度总和，取慢 rank，包含设备通信等待。Denoising stage 还包含 DiT 加载，不能作为纯计算时间。

| 组 | 首次生成 s | 后三次中位 s | 后三次范围 s | 标准差 s | Transformer CUDA span 中位 s | 峰值 allocated GiB | 两卡 KV 发送 GB/次 |
|---|---:|---:|---|---:|---:|---:|---:|
| A | 61.44 | 60.69 | 54.34–64.64 | 4.24 | 15.65 | 36.74 | 16.01 |
| B | 67.36 | 64.87 | 60.31–65.74 | 2.38 | 17.94 | 36.74 | 19.53 |
| C | 69.88 | 61.77 | 59.76–64.00 | 1.73 | 17.61 | 36.74 | 15.96 |

A = Naive Ring；B = 拟合静态 Ring；C = 同布局 Demand。
峰值是 worker 生命周期 allocator 高水位，未逐次 reset；不是显卡总 in-use 显存。KV 数字来自独立观测，使用十进制 GB，不包括 FSDP、selector allreduce 或诊断哈希 collectives。

本次 B 的 transformer span 明显更长。C 的完整时间与 A 的重载/解码波动范围重叠，不能据三个后续样本声称稳定加速。

## 两卡 attention 数值与 microbenchmark

2-head 冒烟全部路径通过独立 masked full-attention oracle，最大绝对误差 0.001953125，无 NCCL 错误/挂起。
56-head 覆盖 `(prompt,step,layer)=(8,2,20),(9,2,20),(8,0,0),(9,3,49)`，全部校验通过，最大绝对误差 0.00390625。
主测试点慢 rank 中位时间：naive sparse Ring 4.75 ms，static Ring 8.69 ms，static Demand 8.60 ms。
这是**真实 NCCL + 真实捕获 mask + 合成 QKV**；包含每次调用的 QKV/输出重分布，不是模型 persistent ownership 的端到端测量。

## 为什么静态布局没有收益

训练侧 planner 在 union、假设 50 TFLOP/s / 25 GB/s 下选择 `prefix_spread=1, audio_replicas=1, video_policy=contiguous`，离线 `phase1_go=true`。这些假设不是硬件标定。

- 实际 token owner：A 为 1396/1395，B/C 为 1703/1088。固定等 tile 数不等于等 token 数，尾 tile 较短；集中 prefix 后，rank 0 的 residual/FFN 工作量增加。
- Ring 需按最大 owner shard 补齐；B 的实际 KV 转发从 A 的 16.01 GB 增至 19.53 GB。
- Demand 将 C 降至 15.96 GB，基本回到 A 水平。运行时按 token 发送所有 heads；离线 per-head demand 的收益不能直接当成物理流量收益。
- B/C 每次另有 26,406,912 bytes hidden-state 迁移，两 rank 各八次迁移调用，包含每个 forward 开头搬入和结尾恢复；逐层 attention QKV 重分布为零。
- 独立 profiler 中，每 rank NCCL all-gather 累计 kernel 持续时间约 11–12 秒，FSDP/加载仍是重要成本。累计 duration 可重叠，不等于墙钟、GPU idle 或独立链路带宽；SendRecv 同时包含加载及其他 P2P，未冒充纯 KV 带宽。

## 输出一致性：未通过严格验收

12 个生成文件均有 256×256、124 帧视频和非空有限音频，最终 video/audio latent 均有限。代表帧是符合 prompt 的骑车人场景。
但 A 自身在相同 seed 下出现明显 latent 差异，因此 A/B/C 差异不能全部归因于 policy，不能把这些视频当作已验证严格等价的回归 reference。

独立 A 重复诊断：两次初始 video/audio 噪声、conditioning，以及两 rank 各自首层 Q/K 哈希相同；首层 selector score 哈希不同。mask 从 step 0 / layer 7 开始分叉，193/200 条最终 mask 记录不同。
独立固定-input pooling 控制：BF16 默认模式 12/12 次结果发生约 2.98e-8 的变化，确定性算法模式为 0/12；FP32 控制也复现同类现象。这证实当前 pooling 具有数值非确定性，与实际 score 漂移吻合，**未证明它是全部漂移的唯一来源，也没有用确定性全模型重新验收**。
当前核心 attention/model 实现未被修改，性能表反映上述原生行为。下一轮应先固定 pooling/selector 可复现性，再验证模型级 policy parity；布局目标还需计入 FFN/token 偏斜及实际 all-head transport，暂不据此扩展到四卡。

## 归档与本机原始结果

[归档完整性核对](ARCHIVE_CHECK.md) 列出了已记录项目、源码哈希验证和仍未采集/未验收的边界。

本目录保存可提交的精简报告与 CSV/JSON 证据。视频、原始 packed masks、完整 NPY/NPZ、运行日志和大型 profiler trace 保留在被 Git 忽略的 `outputs/`，不会随 clone 分发。

- [性能表](comparison.csv)、[完整性能与差异](comparison.json)、[逐对 latent 差异](pairwise_latent_consistency.json)。
- [Microbenchmark](microbenchmark_summary.json)、[通信 payload](payload_summary.json)、[owner 分布](ownership_analysis.json)、[profiler 汇总](profiler_summary.json)。
- [A 重复诊断](repeat_diagnostic.json)、[pooling 控制](pooling_reproducibility.json)、[音视频基本检查](media_sanity.json)。
- [模型身份](model_identity.json)、[硬件/驱动](hardware.json)、[实际配置](run_configs.json)、[精确命令](commands.json)、[原始小测量与源码快照](as_run/)。
- [环境](environment.json)、[十条 prompt](prompts.json)、[trace provenance](trace_provenance.json)、[训练侧选择](phase1_selection.json)。
- [静态布局摘录](selected_static_layout.json) 去除了本轮未使用的 adaptive schedules；原始完整布局的哈希见 [源文件清单](source_artifacts.json)。
- [执行记录与近期 commit 功能](RUN_RECORD.md)；[trace 采集操作说明](../../../inference/h3-sparse-ring-experiments.md)。

本机原始结果位置（其他 checkout 不保证存在）：

- `outputs/h3-abc-seed1000/`：A/B/C 视频、日志、latent、独立观测及每 rank profiler trace。
- `outputs/h3-new-trace/converted/`：完整新 trace；`raw/`：packed masks；精确采集/转换代码快照也保留在该父目录。
- `outputs/h3-plan-2/`：完整 fitted layout、离线 CSV、provenance 和 HTML 报告。
- `outputs/h3-ring-smoke-2gpu/`、`outputs/h3-ring-bench-2gpu/`、`outputs/h3-ring-bench-p*`：两卡 oracle 与 microbenchmark 原始结果。

模型位于 `/tmp/h3-a6000-full/model`，推理依赖 overlay 位于 `/tmp/h3-a6000-full/deps`，分析依赖位于 `/tmp/h3-analysis-deps`。这些本机依赖与模型未收入 Git。复现需重新准备模型/环境，并为每次运行选择新的输出目录。

## 检查

本地布局/通信/转换正确性测试：56 passed。转换器后续暴露 CLI 形状参数后，8 项转换测试复测通过。pre-commit 与 git diff --check 通过；脚本和 local tests 的 lint 排除遵守仓库配置，没有绕过。
全部实验进程已退出，两张 GPU 显存占用恢复为零。本目录提供 Git 可追踪归档；原始大文件未收入 Git，未上传外部服务。
