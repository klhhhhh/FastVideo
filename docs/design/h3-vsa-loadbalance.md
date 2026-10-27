# H3 VSA 布局与通信成本实验

实现位于 `scripts/analysis/h3_loadbalance/`，结果写入 CLI 的 `--output` 目录。
这是离线布局模拟器及单 GPU 数值回放，不接入 `parallel_state.py`，不改变 H3 推理行为。

## 复现

在 FastVideo 仓库根目录运行，使用安装了 NumPy、PyTorch、Matplotlib、pytest 的环境：

```bash
python -m pytest tests/local_tests/h3_loadbalance/test_policies.py -q
python scripts/analysis/h3_loadbalance/run_experiments.py \
  --source /home/kun/output_h3_vsa/predictability-20261007 \
  --output /home/kun/output_loadbalance_policy
python scripts/analysis/h3_loadbalance/validate_baseline.py \
  --output /home/kun/output_loadbalance_policy
python scripts/analysis/h3_loadbalance/cuda_replay.py \
  --output /home/kun/output_loadbalance_policy
python scripts/analysis/h3_loadbalance/build_report.py \
  --output /home/kun/output_loadbalance_policy
```

baseline 校验与 CUDA 回放脚本使用同一个默认 source；更换输入目录时也需要传入 `--source`。
报告生成器只读取 output，不需要 source。输入是已有捕获文件，不下载模型。
`experiment.json` 保存输入 SHA256、代码 SHA256、版本和原 trace provenance。

## 输入与分割

真实 H3 trace 包含 10 个 prompt、4 个 denoising forward、50 层、56 个 head。
每个样本有 8 个 prefix tile（text 1、audio 7）及 40 个 video tile，坐标为 10×2×2。
video 为 2368 token；每个 video query tile 选择 4 个 video KV tile。
text/audio Q 对全部 KV 可见，video Q 对全部 prefix KV 可见。
`layout_XX.npz` 的 packed-token→tile 映射用于计费，不把尾 tile 当作完整 64 token。

训练只用 prompt 0–7，报告用未参与训练的 prompt 8、9。所有 4 step、50 层、56 head 均参与评估。
原 mask 不作任何修改。每个配置有 11200 个 head 样本；汇总计算不均衡时先把同一层的
56 个 head 合并到 rank，再计算 max/mean。因此这个 rho 与旧报告逐 head 的 p95 不同。
没有不同 seed 的泛化验证，也没有把这些相关 head 当独立统计重复。

## 布局策略

- `naive_packed`：按 packed token 序列等长 contiguous 分配 Q/K/V，末 rank 可略短。
- `rr_temporal_*`：prefix token round-robin；video 按 t、h、w 顺序等 tile 数切块。
  rank 边界允许切开一个 temporal plane。
- `rr_spatial_*`：prefix round-robin；video 按 h、w、t 顺序等 tile 数切块。
  G=8 时每个 spatial column 会分成两个时间段。
- `rr_3d_*`：prefix round-robin；video 按 G=2/4/8 分成 2×1×1、2×2×1、2×2×2 个块。
- `weighted_3d_textrep`：先用固定 10% video density 估算 rank video 工作量，再将每个
  prefix token 分配给当前预计工作量最小的 rank；不读取 held-out mask。
- `predict_static`：将训练 prompt 的全部 step/head/layer 平均得到 query-specific prior，
  从 3D base 开始最多接受两次 video tile swap，整次推理沿用同一布局。
- `predict_layer`：每层单独平均训练 prompt/step/head 的 prior，从 3D base 开始最多两次 swap。
- `predict_minimal`：从训练得到的 static 布局开始，逐层在上一布局上最多两次 swap，
  目标包含迁移惩罚；学到的 50 层映射在 4 step 重复使用，评估包含 step 边界迁移。

每次 swap 保持每 rank tile 数；token 数可因尾 tile 而变化。Q、K、V 的 canonical owner
始终共同移动。prefix 分配是确定性、静态的；预测策略只交换 video tile。

预测训练目标为 `rho_expected + B_video_demand/B0 + gamma*B_hidden_migration/B0`，
`B0=(G-1)*N*512`，minimal 的 gamma=1，其余为 0。video demand 使用
`1-product(1-P(q,k))` 估算 rank 对 KV tile 的去重需求，**假设不同 query 的边独立**。
这是可解释的启发式，未声称精确建模联合 topology；计算量的期望为精确线性求和。
prefix 全局需求、text refresh 对候选 swap 恒定，故不放入训练目标。
held-out 评估直接用真实 mask 计算精确去重字节，不使用训练时的独立性近似。
未调优 gamma，也没有针对 held-out 选择最优预测超参数。

## KV residency 与通信

`*_shard` 不复制 KV；`*_textrep` 复制 text 到所有 rank，audio/video shard。
`audio_groups_2/4/8` 同时复制 text，并让每个 audio group **合计保存一份 audio KV**。
例如 8 rank、2 group：每组 4 rank，组内 shard，组间两份副本。
同组并不代表远程读取免费。每 token 在各组的相同 local-rank 有一份副本，包含原 owner。

每条传输按 BF16 K+V、head_dim=128 计费：每 token 每 head 512 byte。
每个 layer/step 刷新全部显式副本，不跨层/step 重用 KV。
传输为逻辑 unicast；由 canonical owner 向各目标发送，不使用 multicast、转发或最优副本源路由。
同时输出 send/receive，避免只看接收字节漏掉 sender bottleneck。

评估两种缓存调度：

- `union`：同层 prefix/video query 共享所有已获取 KV，每目的 rank 每 token 最多获取一次。
- `separate`：prefix/video 分两个阶段，各自去重，阶段间丢弃按需获取的远端 KV，保留 owner
  和显式副本。一个 token 最多出现两次 demand；副本仍只 refresh 一次。

两种模型均在下一 layer/step 冷启动。union 是有利的字节量下界；separate 代表一种受限调度，
不是已证明的真实 kernel 行为。若全部 rank 都有 dense prefix Q，则 union 中每 rank 都需要
全部 KV，因此 `demand+refresh=(G-1)*N*512` 对布局/复制方案不变。
这时 video locality 的通信收益只有在调度、缓存窗口等额外约束下才显现。

`resident_kv_peak_bytes` 仅计 owner 与显式副本，`extra_kv_peak_bytes` 仅计新增副本，
均含 56 head。它们不是 GPU 峰值显存：不包含 fetched KV、Q、输出、临时 attention score
或模型权重。若 union 实现选择全量 materialize，dense-prefix rank 的 KV 至少达到全序列大小；
流式实现可能减少 buffer，但这里未建模缓存淘汰与 chunk 调度。

## Layout transition 与 latency proxy

记录 initial packed→首层布局和所有 199 个后续层/step 边界。
固定布局没有后续迁移；如果输入本来就按目标布局分配，initial 成本也可以省去。
每个转移 token 默认搬运一个 BF16 hidden-state stream，5376×2 byte；这是乐观 payload，
不包含额外 residual/buffer。参数扫描同时使用 4 倍 payload（数值等于 56×128 的 Q/K/V
三份 BF16 tensor），用于检验更重迁移的影响，不把 hidden 与 projected QKV 重复计费。
具体 runtime 如何搬运、恢复顺序、执行 FFN 尚未实现。

每层先合并 56 head 后，定义：

```text
compute_g = token_pairs_g * (4*128) / effective_attention_FLOPs_per_second
comm_g = max(send_bytes_g, receive_bytes_g) / full_duplex_bandwidth
layer_time = max_g(compute_g + comm_g - overlap*min(compute_g, comm_g))
migration_time = max_g(migration_send_g, migration_receive_g) / bandwidth
proxy = sum(layer_time + migration_time) + modeled_sync_time
```

扫描 effective compute 10/50/200 TFLOP/s、链路 10/25/100 GB/s、overlap 0/0.5/1、
每层及每次非空迁移同步 0/5 微秒、迁移 payload 倍数 1/4。每配置 216 个 proxy 组合。
带宽、吞吐和同步开销均是**假设参数，不是 GB10 实测标定**。
计算只含 attention 的 QK、PV，不含 softmax、FFN、投影、tile padding、kernel 启动、调度
及 gather/scatter 本地开销。分阶段缓存的时间估算仍是总量 proxy，没有精确模拟阶段串行和
依赖图。因此它只能指出 trade-off 的敏感性，不能换算成真实端到端加速比。

## 验证与输出

`test_policies.py` 用独立逐 token 枚举检查工作量、去重需求、收发流量、副本刷新，并检查
group residency、迁移和布局容量。`validate_baseline.py` 在全部 10 prompt × 3 rank 设置下
逐项匹配旧 trace 分析保存的 rank 工作量、send、receive 和总 demand。

`cuda_replay.py` 在一个真实 GPU 上顺序执行 8 个虚拟 rank，使用真实 mask 和固定随机 Q/K/V，
分别 gather 所需 KV、计算 masked attention、scatter 输出，对照未分片的 masked attention。
覆盖 held-out 两个 prompt，step 2 / layer 20 / head 0、17 的全部 8-rank 策略。
这是布局语义和 demand 校验：不是原模型激活，不是 VSA kernel 性能测试，不验证视频质量，
不启动 NCCL，也没有真实多卡通信。

输出包括 `summary.csv`、`per_rank.csv`、`latency_sweep.csv`、`assignments.json`、
`experiment.json`、`baseline_validation.json`、`cuda_validation.json`、图表和 `report.html`。
报告由结果生成，避免将手写数字作为实验事实。
