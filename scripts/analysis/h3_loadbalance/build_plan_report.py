"""Generate the four-phase report solely from experiment artifacts."""
import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean


def table(rows, fields):
    def cell(value):
        if isinstance(value,float):
            value=f'{value:.5g}'
        return html.escape(str(value))
    return '<table><tr>'+''.join('<th>'+cell(k)+'</th>' for k in fields)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+cell(r.get(k,''))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'


def grouped(rows, fields, split='heldout', group='policy'):
    names=list(dict.fromkeys(r[group] for r in rows))
    result=[]
    for name in names:
        values=[r for r in rows if r[group]==name and r.get('split',split)==split]
        if values:
            result.append(dict(policy=name,**{f:mean(float(r[f]) for r in values) for f in fields}))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output
    def read(name):
        with (root/name).open() as f:
            return list(csv.DictReader(f))
    selection=json.loads((root/'phase1_selection.json').read_text());cache=selection['cache']
    phase1=grouped(read('phase1.csv'),['rho',cache+'_bytes',cache+'_proxy_ms',cache+'_audio_demand_bytes','refresh_bytes','extra_kv_bytes'])
    phase2=grouped(read('phase2.csv'),['rho',cache+'_video_demand_bytes','local_selected_edge_ratio',
                                     'remote_query_rank_fanout','temporal_long_range_dedup_bytes',cache+'_proxy_ms'])
    pred=grouped(read('phase3_prefetch.csv'),['recall','precision','prefetch_efficiency','wasted_prefetch_bytes',
                                          'residual_remote_bytes','predicted_remote_query_rank_fanout'],group='budget')
    adaptive=grouped(read('phase3_adaptation.csv'),['transition_migration_bytes',cache+'_bytes',cache+'_proxy_ms'])
    hardware=grouped(read('phase3_hardware.csv'),[cache+'_proxy_ms'])
    boundary=[]
    for kind in ['nvlink','pcie_same_numa','pcie_cross_numa','custom']:
        candidates=[r for r in hardware if r['policy'].startswith(kind+'_')]
        if candidates:
            best=min(candidates,key=lambda r:r[cache+'_proxy_ms'])
            boundary.append(dict(topology=kind,best_evaluated_policy=best['policy'],proxy_ms=best[cache+'_proxy_ms']))
    (root/'phase3_boundary.json').write_text(json.dumps(boundary,indent=2))
    sections=[]
    for title,rows in [('Phase 1 — held-out joint sweep',phase1),('Phase 2 — held-out video locality',phase2),
                       ('Phase 3 — prediction and video-phase prefetch',pred),('Phase 3 — adaptation',adaptive),
                       ('Phase 3 — hardware sensitivity boundary',boundary)]:
        sections.append('<h2>'+title+'</h2>'+table(rows,list(rows[0])))
    runtime=[]
    for path in sorted(root.glob('*/distributed_attention.json')):
        r=json.loads(path.read_text())
        runtime.append(dict(artifact=str(path.relative_to(root)),world_size=r['world_size'],hardware=r['hardware'],
                            gpu_performance_measured=r['gpu_performance_measured'],model_inference=r['model_inference']))
    result='''<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 four-phase experiments</title>
<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>H3 four-phase experiments</h1>
<p>Train prompts 0–7 select policies; held-out prompts 8–9 evaluate them. Offline times are attention proxies,
not real model speedups. GB are decimal. Each offline row includes 4 steps × 50 layers × 56 heads.</p>
<h2>Phase 1 selection</h2><pre>'''+html.escape(json.dumps(selection,indent=2))+'''</pre>
<img src="phase1_pareto.png" alt="Train-only operating points">
<p>The top three are ranked under the declared compute/bandwidth/cache assumptions. The Pareto frontier is a separate
multiobjective result. Later phases remain exploratory if the Phase 1 go/no-go is false.</p>'''+''.join(sections)+'''
<h2>Phase 4 validation evidence</h2>
<p>The runtime uses fused Triton sparse accumulation for CUDA BF16/FP16 and a portable CPU/FP32 oracle. Four-process Gloo tests exercise communication and correctness only. One-GPU replay is not multi-GPU
validation. Real generation comparisons are produced separately by benchmark_inference.py on a multi-GPU host.</p>'''
    if runtime:
        result+=table(runtime,list(runtime[0]))
    result+='''<h2>Modeling and runtime differences</h2><ul>
<li>Logical demand is per-head deduplicated unicast. Runtime transfers all heads of a token together, so payloads can be larger.</li>
<li>Classic Ring forwards padded owner buffers. Demand mode is a direct cyclic P2P exchange, not a forwarding Ring.</li>
<li>Model inference persists hidden ownership across blocks, updates RoPE/AdaLN row indices, and restores packed
ownership once per transformer forward. This adds step-boundary restoration absent from the offline proxy.
The standalone attention microbenchmark instead includes per-call QKV/output redistribution; these are distinct measurements.</li>
<li>Runtime replica refresh is fused with union demand in the same transfer. The separate-cache proxy is not its execution schedule.</li>
<li>Prefetch never changes the observed mask; misses are fetched and unused prefetched tokens are masked out.</li>
<li>Prefetch diagnostics isolate the video phase; dense prefix queries can eliminate their benefit under union caching.</li>
<li>Hardware matrices are assumed seconds/byte, not calibration. Adaptive schedules and drift triggers use train priors,
not current held-out masks.</li>
</ul><h2>Artifacts</h2><ul>'''
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name!='plan_report.html':
            result+='<li><a href="'+html.escape(path.name)+'">'+html.escape(path.name)+'</a></li>'
    result+='</ul></html>'
    (root/'plan_report.html').write_text(result)
    print(root/'plan_report.html')


if __name__=='__main__':
    main()
