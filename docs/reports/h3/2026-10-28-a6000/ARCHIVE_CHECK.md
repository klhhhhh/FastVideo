# 报告归档完整性核对

本次核对后，实验的主要配置、过程、原始数值、失败结果和可复现入口均有记录。核对没有重新运行完整模型，也没有将未通过的输出一致性改写为通过。

| 需要记录的内容 | Git 内记录 |
| --- | --- |
| 模型、revision、导出声明、schedule、下载 ETag | [model_identity.json](model_identity.json)；67 个本地下载 metadata 均指向 `5ea076f35b84da4c3c82217112fa733d8eea2ae1` |
| GPU 型号、显存、驱动、PCIe/NUMA 拓扑 | [hardware.json](hardware.json)、[environment.json](environment.json) |
| prompt、seed、训练/held-out 划分、完整实际参数 | [prompts.json](prompts.json)、[run_configs.json](run_configs.json)、[trace_provenance.json](trace_provenance.json) |
| 精确 A/B/C、观测、profiler argv 与路径 | [commands.json](commands.json)、[RUN_RECORD.md](RUN_RECORD.md) |
| 当时运行的未提交源码，而非只留一个 Git HEAD | [as_run/](as_run/)、[诊断快照](repeat_diagnostic.py.snapshot)；源码 SHA256 与对应 run config/provenance 匹配 |
| 每次、每 rank 的时间与实际 sigma | `as_run/*measurements_rank_*.jsonl`、[comparison.json](comparison.json) |
| 数值 oracle、性能、KV/迁移与 profiler 汇总 | [REPORT.md](REPORT.md) 和同目录 CSV/JSON |
| OOM、sample metadata 恢复、依赖缺失及处理 | [RUN_RECORD.md](RUN_RECORD.md) 和 [trace_provenance.json](trace_provenance.json) |
| 相同输入的 selector 漂移、pooling 控制、未完成验收 | [repeat_diagnostic.json](repeat_diagnostic.json)、[pooling_reproducibility.json](pooling_reproducibility.json) |
| 来源与归档完整性 | [source_artifacts.json](source_artifacts.json)、[bundle_manifest.json](bundle_manifest.json)、[archive_audit.json](archive_audit.json) |

## 明确保留的边界

- 模型导出声明中的四卡/Blackwell 推荐配置，不是本轮实际参数；本轮为两张 A6000、pure sparse Ring/Triton。实际设置以 `run_configs.json` 和源码快照为准。
- checkpoint content SHA256 是导出声明；下载 revision/ETag 来自本地 HF metadata，未将整套权重独立重哈希后冒称通过。
- `environment.json` 是运行后的独立 Python 环境快照，没有采集每次 worker 的全部环境变量或安装 wheel 哈希锁定文件。
- 原始完整 mask、NPY/NPZ、视频、latent、日志和大型 profiler traces 仍只在本机 ignored outputs。仓库保留来源哈希和关键小证据，单独 clone 仓库不能恢复那些大文件。
- 严格模型输出一致性仍未通过；没有确定性全模型复验，也没有更多硬件拓扑/四卡/其他分辨率的性能结论。
