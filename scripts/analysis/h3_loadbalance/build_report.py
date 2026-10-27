"""Generate a standalone HTML report and figures from completed experiment artifacts."""
import argparse
from collections import defaultdict, Counter
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    summaries=list(csv.DictReader((root/'summary.csv').open()))
    sweep=list(csv.DictReader((root/'latency_sweep.csv').open()))
    cuda=json.loads((root/'cuda_validation.json').read_text())
    validation=json.loads((root/'baseline_validation.json').read_text())
    experiment=json.loads((root/'experiment.json').read_text())
    grouped=defaultdict(list)
    for r in summaries:
        if r['ranks']=='8':
            grouped[r['policy']].append(r)
    means={k:{field:float(np.mean([float(r[field]) for r in rows])) for field in rows[0] if field not in ['policy','prompt','ranks']} for k,rows in grouped.items()}
    labels=list(means)
    fig, axes=plt.subplots(1,2,figsize=(14,7))
    y=np.arange(len(labels))
    axes[0].barh(y,[means[k]['rho_mean'] for k in labels])
    axes[0].set_yticks(y,labels)
    axes[0].set_xlabel('Mean per-layer max/mean work (56 heads combined)')
    axes[0].invert_yaxis()
    axes[1].barh(y-.18,[means[k]['union_total_bytes']/1e9 for k in labels],height=.35,label='union')
    axes[1].barh(y+.18,[means[k]['separate_total_bytes']/1e9 for k in labels],height=.35,label='separate')
    axes[1].set_yticks(y,[])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Demand + refresh, GB / 4-step trace')
    axes[1].legend()
    fig.suptitle('8 logical ranks; mean of held-out prompts 8 and 9')
    fig.tight_layout()
    fig.savefig(root/'balance_traffic.png',dpi=160)
    plt.close(fig)
    rows=[]
    for name,m in means.items():
        rows.append([name,f"{m['rho_mean']:.4f}",f"{m['union_total_bytes']/1e9:.3f}",f"{m['separate_total_bytes']/1e9:.3f}",f"{m['refresh_bytes']/1e9:.3f}",f"{m['extra_kv_peak_bytes']/1e6:.3f}",f"{m['transition_hidden_bytes']/1e6:.3f}"])
    main_table=table(['策略','层 rho 均值','union GB','separate GB','refresh GB','最大 rank 额外 KV MB','层间 hidden 迁移 MB'],rows)
    aggregated=defaultdict(list)
    for r in sweep:
        if r['ranks']=='8':
            key=tuple(r[f] for f in ['cache','tflops','bandwidth_GBs','overlap','sync_us','migration_payload_multiplier','policy'])
            aggregated[key].append(float(r['proxy_ms']))
    avg={k:float(np.mean(v)) for k,v in aggregated.items()}
    selected=['naive_packed','rr_3d_textrep','weighted_3d_textrep','audio_groups_4','audio_groups_8','predict_static','predict_layer','predict_minimal']
    demo=[]
    for cache in ['union','separate']:
        for flops,bw in [('10','100'),('200','10')]:
            for name in selected:
                key=(cache,flops,bw,'0.0','0.0','1',name)
                demo.append([cache,flops,bw,name,f'{avg[key]:.3f}'])
    latency_table=table(['缓存','假设 TFLOP/s','假设 GB/s','策略','proxy ms / 4-step trace'],demo)
    regimes=defaultdict(dict)
    for key,value in avg.items():
        regimes[key[:-1]][key[-1]]=value
    wins=Counter()
    for key,policies in regimes.items():
        best=min(policies.values())
        # Share ties explicitly; these are grid counts, not probabilities.
        tied=tuple(k for k,v in policies.items() if abs(v-best)<1e-9)
        wins[' / '.join(tied)]+=1
    winners=table(['8-rank 参数网格最小 proxy 策略（并列完整列出）','网格数量'],wins.most_common())
    comparisons=[]
    for cache in ['union','separate']:
        improvements=[]
        for regime,policies in regimes.items():
            if regime[0]==cache:
                improvements.append(policies['naive_packed']/policies['weighted_3d_textrep'])
        comparisons.append([cache,f'{min(improvements):.3f}',f'{max(improvements):.3f}'])
    ratios=table(['缓存','weighted 相对 naive 最小 proxy 比值','最大 proxy 比值'],comparisons)
    naive,base,weighted=[means[k] for k in ['naive_packed','rr_3d_textrep','weighted_3d_textrep']]
    maxerror=max(r['max_abs_error'] for r in cuda['checks'])
    report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>H3 VSA load balance 与通信 trade-off</title>
<style>body{{max-width:1300px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#182536;background:#f7f9fc}}h1,h2{{color:#153d62}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px;margin:20px 0}}td,th{{padding:8px;border:1px solid #d9e1ea;text-align:left}}th{{background:#e6eff7}}img{{max-width:100%}}.note{{padding:18px;background:#e7f1fc;border-left:4px solid #2762a2}}code{{background:#e9edf2;padding:2px 5px}}</style>
<h1>H3 VSA：静态负载均衡与 KV 搬运的取舍</h1>
<p>仓库实现：<code>scripts/analysis/h3_loadbalance/</code>；实现文档：<code>docs/design/h3-vsa-loadbalance.md</code>。实验结果由脚本自动汇总。</p>
<div class="note"><b>执行方式：</b>真实单卡 H3 trace 上的 CPU 离线逻辑 2/4/8-rank 模拟，另在 {html.escape(cuda['device'])} 一个 GPU 上顺序回放 8 个虚拟 rank。没有真实多卡通信，也没有端到端推理加速测量。表中的 latency 全部是假设吞吐、链路、overlap 和同步开销下的 attention proxy。</div>
<h2>主要结果</h2>
<p>两个 held-out prompt 的 8-rank 层 rho 均值：naive <b>{naive['rho_mean']:.3f}</b>，prefix round-robin + 3D + text replica <b>{base['rho_mean']:.3f}</b>，weighted prefix <b>{weighted['rho_mean']:.3f}</b>。
但 union 总通信从 <b>{naive['union_total_bytes']/1e9:.2f} GB</b> 增到 <b>{base['union_total_bytes']/1e9:.2f} GB</b>。均衡不等于更快。</p>
<p>原因是每个 rank 获得 dense prefix Q 后，每个 rank 都需要全量 KV。整层充分去重时，所有这些策略的 demand+refresh 都等于 (G−1)×N×512 byte/head。
text/audio 副本只能改变传输时机与驻留内存，不能降低这个总量。separate 模型中显式副本避免跨阶段重复获取，因此 replication 有节省；这个结论依赖缓存调度。</p>
<img src="balance_traffic.png" alt="策略的计算不均衡与通信字节对比">
<h2>每个实验如何做</h2>
<ol><li><b>静态 modality balance：</b>保留原始 sparse mask，改变 canonical Q/K/V owner。对照 packed naive、prefix round-robin、按固定 10% density 估算的 weighted prefix；分别尝试 temporal、spatial、3D video 分区。</li>
<li><b>Residency：</b>比较 shard、text 全复制、audio 2/4/G group 副本。组内 audio 是 shard，不假设同组通信免费；每 layer/step 重新刷新 K 和 V。发送源固定 canonical owner，同时计 send/receive。</li>
<li><b>预测 refinement：</b>只用 prompt 0–7 的真实 mask 估计 query-specific prior；比较跨层固定 prior、逐层 prior、带迁移惩罚的逐层 prior，每轮最多两次 tile swap。训练 dedup 使用独立边近似，评估使用精确真实 mask。prompt 8/9 未参与训练。</li>
<li><b>迁移与 latency：</b>记录 initial 重排及后续 199 个边界；按单份 BF16 hidden state 计费并扫描 4 倍 payload。比较 10/50/200 TFLOP/s、10/25/100 GB/s、overlap 0/0.5/1、同步 0/5 微秒。每层先合并所有 head，再求 rank 最大计算/通信路径。</li>
<li><b>单 GPU 数值回放：</b>真实 mask 配固定随机 FP32 Q/K/V，在一个 GPU 上逐个执行逻辑 rank，gather KV、attention、scatter 输出，与未分片 masked attention 比较。固定 step 2、layer 20、head 0/17，共 {len(cuda['checks'])} 项通过；最大绝对误差 {maxerror:.3g}。没有原模型 QKV，不能推断视频质量或实际 VSA kernel 性能。</li></ol>
<h2>8-rank 总量（两个 held-out prompt 均值）</h2>
<p>GB/MB 均为十进制；总量覆盖 4 step×50 layer×56 head。rho 是每层先聚合 56 head 后的 max/mean，再对 200 层取均值。额外 KV 仅为显式副本，不是 GPU 峰值显存。</p>{main_table}
<p>predict_static 若没有接受 swap，会与结构 base 相同；predict_minimal 的当前 gamma=1 可拒绝全部动态调整。这是训练启发式的实际结果，不代表所有预测优化都无效。predict_layer 的迁移量揭示了逐层重排的真实附加负担；没有因为 rho 接近就忽略它。</p>
<h2>通信/计算取舍</h2>
<p>proxy = Σ max_rank(compute+comm−overlap×min(compute,comm)) + migration + sync；comm 用 max(send,receive)/bandwidth，假设 full duplex。以下两个参数区间仅为敏感性示例，overlap=0、sync=0、migration payload=1。数值覆盖整份 4-step attention trace，<b>不是模型生成耗时</b>。</p>{latency_table}
<p>以下比值为 naive proxy / weighted proxy，大于 1 表示该假设下 weighted 更优。参数网格不是硬件性能分布，也不构成实测 speedup：</p>{ratios}{winners}
<h2>验证、来源与限制</h2>
<ul><li>{len(summaries)} 个策略/prompt/rank 配置，{len(sweep)} 条 latency 参数组合；训练 8 prompt，held-out 2 prompt，seed 固定 1000。所有 4 step、50 layer、56 head 均纳入离线评估，video grid 10×2×2、40 tile，其中尾 tile 长度 16。</li>
<li>{len(validation)} 组全 prompt×rank 原基线的 token-pair work、send、receive、dedup demand 与之前独立保存的 trace 分析逐项一致。另有逐 token oracle 和 residency/迁移单元测试，见 tests.log。</li>
<li>固定布局没有层间迁移。当前 proxy 包括从 naive 输入分布到首层布局的一次转换；如果上游直接按布局生产 hidden states，可省去这笔成本。</li>
<li>当前只估算 attention QK/PV，未计 FFN、投影、softmax、padding、kernel 启动、本地 gather/scatter、物理网络拓扑、collective 算法及实际调度依赖。separate 的阶段时间也不是精确执行模拟。</li>
<li>仅计算显式驻留 KV；union 全量 materialize 时 fetched KV 另占空间。没有在 production runtime 实现 layout 切换或跨层 KV reuse。</li>
<li>建议先验证静态 weighted prefix + video 分块的真实执行成本，再决定复制和动态 refinement；在当前 trace 上继续追求更低 rho 的空间已经较小。</li></ul>
<h2>可复现产物</h2><p>实现文档给出完整命令。运行 provenance：{html.escape(experiment['git_head'])}，未提交实验代码另以 SHA256 记录。</p>
<ul>''' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['summary.csv','per_rank.csv','latency_sweep.csv','assignments.json','experiment.json','baseline_validation.json','cuda_validation.json','tests.log','analyze.log']) + '</ul></html>'
    (root/'report.html').write_text(report)
    print('Wrote',root/'report.html')
    for name in ['naive_packed','rr_3d_textrep','weighted_3d_textrep','predict_layer']:
        print(name,means[name])
    print('proxy ranges',comparisons)


if __name__=='__main__':
    main()
