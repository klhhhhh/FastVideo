"""Four-phase experiment inputs: train-selected layouts and held-out diagnostics.

Phase 4 is executed by benchmark_ring.py on real distributed processes. No
simulator number is labeled as measured latency. All byte counts are logical
unicast; hardware matrices are user-supplied assumptions in seconds/byte.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from policies import HIDDEN_BYTES, KV_BYTES, Layout, evaluate, migration, owners, video_layout


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pareto(rows, fields):
    return [r for r in rows if not any(
        all(s[f] <= r[f] for f in fields) and any(s[f] < r[f] for f in fields) for s in rows)]


def topology_matrix(ranks, kind):
    # Illustrative costs, deliberately not calibration claims.
    bw = np.full((ranks, ranks), 100. if kind == 'nvlink' else 25.)
    if kind == 'pcie_cross_numa':
        side = np.arange(ranks) >= ranks // 2
        bw[side[:, None] != side[None, :]] = 10.
    cost = 1 / (bw * 1e9)
    np.fill_diagonal(cost, 0.)
    return cost


def graph_layout(prob, sizes, ranks, base, method, cost=None, swaps=2):
    """Capacity-constrained graph placement; deterministic, train-only weights."""
    cost = topology_matrix(ranks, 'nvlink') if cost is None else cost
    weights = prob * sizes[None, :]
    weights = weights + weights.T
    capacity = np.bincount(base, minlength=ranks)
    if method == 'graph':
        # Spectral ordering of the symmetrized weighted graph, then equal bins.
        lap = np.diag(weights.sum(1)) - weights
        _, vectors = np.linalg.eigh(lap)
        order = np.argsort(vectors[:, 1], kind='stable')
        out = np.empty(len(base), int)
        out[order] = np.repeat(np.arange(ranks), capacity)
    elif method == 'greedy':
        out = np.full(len(base), -1, int)
        remaining = capacity.copy()
        for tile in np.argsort(-weights.sum(1), kind='stable'):
            placed = out >= 0
            scores = np.array([np.sum(weights[tile, placed] * cost[g, out[placed]])
                               if remaining[g] else np.inf for g in range(ranks)])
            g = int(scores.argmin())
            out[tile] = g
            remaining[g] -= 1
    else:
        out = base.copy()
    def objective(v):
        # Expected edge bytes, rather than claiming exact expected dedup.
        return np.sum(weights * cost[v[:, None], v[None, :]])
    for _ in range(swaps):
        best, chosen = objective(out), None
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                if out[a] == out[b]:
                    continue
                trial = out.copy()
                trial[a], trial[b] = trial[b], trial[a]
                value = objective(trial)
                if value < best - 1e-15:
                    best, chosen = value, trial
        if chosen is None:
            break
        out = chosen
    return out


def aggregate(value, field, ranks):
    return value[field].reshape(200, 56, ranks).sum(1)


def measure(masks, layout, ranks, maps, spread, replicas, bandwidth, tflops, weighted=False, cost=None):
    """maps: one video mapping per step/layer, or None for packed baseline."""
    ownmaps = [owners(layout, ranks, None if maps is None else m, weighted, spread)
               for m in ([None]*200 if maps is None else maps)]
    # Evaluate all steps/heads at once for a static layout.
    if all(np.array_equal(ownmaps[0], o) for o in ownmaps):
        value = evaluate(masks, layout, ownmaps[0], ranks, maps is not None, replicas)
    else:
        parts = [evaluate(masks.reshape(200, 56, 40, 40)[i], layout, o, ranks, True, replicas)
                 for i, o in enumerate(ownmaps)]
        value = {k: np.concatenate([p[k] for p in parts]) if parts[0][k].ndim > 1 or
                 k.endswith(('_demand', '_fanout')) else parts[0][k] for k in parts[0]}
    work = aggregate(value, 'work', ranks)
    rho = work.max(1) / work.mean(1)
    old = owners(layout, ranks)
    moves = []
    total_move = []
    migration_times = []
    for own in ownmaps:
        send, recv = migration(layout, old, own, ranks)
        moves.append(np.maximum(send, recv))
        if cost is not None:
            changed = old != own
            transfer = np.zeros((ranks, ranks))
            np.add.at(transfer, (old[changed], own[changed]), HIDDEN_BYTES)
            timeflow = transfer * cost
            migration_times.append(max(timeflow.sum(0).max(), timeflow.sum(1).max()))
        else:
            migration_times.append(np.maximum(send, recv).max()/(bandwidth*1e9))
        total_move.append(send.sum())
        old = own
    moves = np.array(moves)
    row = dict(rho=float(rho.mean()), rho_p95=float(np.quantile(rho, .95)),
               initial_migration_bytes=int(total_move[0]), transition_migration_bytes=int(sum(total_move[1:])),
               extra_kv_bytes=int(value['extra_bytes'].max()*56),
               refresh_bytes=int(value['refresh_recv'].sum()*56*200))
    detail = []
    for mode in ['union', 'separate']:
        send, recv = [aggregate(value, mode+'_'+f, ranks) for f in ['send', 'recv']]
        traffic = np.maximum(send, recv)
        compute = work * 512 / (tflops*1e12)
        comm = traffic / (bandwidth*1e9)
        if cost is not None:
            flows = value[mode+'_flows'].reshape(200, 56, ranks, ranks).sum(1)
            timeflows = flows * cost
            comm = np.maximum(timeflows.sum(1), timeflows.sum(2))
        row[mode+'_bytes'] = int(recv.sum())
        row[mode+'_proxy_ms'] = float((np.max(compute+comm, axis=1).sum()+sum(migration_times))*1000)
        row[mode+'_fanout'] = float(value[mode+'_destination_fanout'].mean())
        row[mode+'_send_imbalance'] = float(np.mean(send.max(1)/np.maximum(send.mean(1), 1)))
        row[mode+'_recv_imbalance'] = float(np.mean(recv.max(1)/np.maximum(recv.mean(1), 1)))
        for modality in ['text', 'audio', 'video']:
            row[mode+'_'+modality+'_demand_bytes'] = int(value[mode+'_'+modality+'_demand'].sum())
        for i in range(200):
            for g in range(ranks):
                detail.append(dict(cache=mode, step=i//50, layer=i%50, rank=g, work=int(work[i,g]),
                                   send_bytes=int(send[i,g]), recv_bytes=int(recv[i,g])))
    return row, detail


def locality(masks, layout, video, ranks):
    m = masks.reshape(-1, 40, 40)
    same = video[:, None] == video[None, :]
    coords = layout.coords
    local = (np.abs(coords[:, None, 0]-coords[None, :, 0]) <= 1)
    column = (coords[:, None, 1:] == coords[None, :, 1:]).all(-1)
    selected = float(m.sum())
    out = dict(local_selected_edge_ratio=float((m*same).sum()/selected),
               remote_selected_edge_ratio=float((m*~same).sum()/selected),
               temporal_local_edge_ratio=float((m*local).sum()/selected),
               same_column_edge_ratio=float((m*column).sum()/selected))
    fanout = np.zeros(m.shape[:2], int)
    longbytes = 0
    for g in range(ranks):
        fanout += (m[..., video == g].any(-1) & (video[None, :] != g))
        need = (m[:, video == g] & ~local[video == g]).any(1)
        longbytes += int((need[:, video != g]*layout.sizes[layout.prefix:][video != g]).sum()*KV_BYTES)
    out['remote_query_rank_fanout'] = float(fanout.mean())
    out['temporal_long_range_dedup_bytes'] = longbytes
    return out


def prefetch_metrics(actual, predicted, sizes, video, ranks):
    """Per-destination video-phase dedup. Prefix demand is deliberately excluded."""
    actual, predicted = [x.reshape(-1, 40, 40) for x in (actual, predicted)]
    totals = dict(useful_prefetch_bytes=0, wasted_prefetch_bytes=0, residual_remote_bytes=0,
                  actual_remote_bytes=0, predicted_remote_bytes=0)
    for g in range(ranks):
        a = actual[:, video == g].any(1) & (video[None] != g)
        p = predicted[:, video == g].any(1) & (video[None] != g)
        for key, need in [('useful_prefetch_bytes', a&p), ('wasted_prefetch_bytes', p&~a),
                          ('residual_remote_bytes', a&~p), ('actual_remote_bytes', a), ('predicted_remote_bytes', p)]:
            totals[key] += int((need*sizes).sum()*KV_BYTES)
    tp = int((actual & predicted).sum())
    totals.update(recall=tp/max(int(actual.sum()),1), precision=tp/max(int(predicted.sum()),1),
                  miss_rate=1-tp/max(int(actual.sum()),1),
                  prefetch_efficiency=totals['useful_prefetch_bytes']/max(totals['predicted_remote_bytes'],1),
                  overfetch_ratio=totals['wasted_prefetch_bytes']/max(totals['actual_remote_bytes'],1))
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/kun/output_h3_vsa/predictability-20261007'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=[2,4,8], default=8)
    parser.add_argument('--bandwidth-gbs', type=float, default=25.)
    parser.add_argument('--tflops', type=float, default=50.)
    parser.add_argument('--cache', choices=['union','separate'], default='union')
    parser.add_argument('--topology-cost', type=Path, help='JSON ranks x ranks matrix, seconds/byte')
    args = parser.parse_args()
    if min(args.bandwidth_gbs, args.tflops) <= 0:
        parser.error('bandwidth and compute must be positive')
    root=args.output; root.mkdir(parents=True, exist_ok=True)
    masks=np.load(args.source/'video_masks.npy', mmap_mode='r')
    if masks.shape != (10,4,50,56,40,40):
        raise ValueError('Expected captured 10-prompt H3 trace')
    layouts=[Layout.load(args.source/f'layout_{p:02d}.npz') for p in range(10)]
    g=args.ranks; base=video_layout(layouts[0], g, '3d')
    degrees=[d for d in [1,2,4,8] if d<=g]
    phase1=[]; detail=[]; configs={}
    def run(phase, name, maps, spread, replicas, prompts, extra=None, cost=None):
        rows=[]
        for p in prompts:
            row, ranks=measure(masks[p], layouts[p], g, maps, spread, replicas,
                               args.bandwidth_gbs,args.tflops,cost=cost)
            row=dict(phase=phase,policy=name,prompt=p,split='train' if p<8 else 'heldout',
                     prefix_spread=spread,audio_replicas=replicas,**row)
            if extra:
                row.update(extra(p))
            rows.append(row)
            if phase==1:
                detail.extend(dict(policy=name,prompt=p,**r) for r in ranks)
        print(phase,name,flush=True)
        return rows
    phase1.extend(run(1,'naive_packed',None,g,1,range(10)))
    for spread in degrees:
        for replicas in degrees:
            name=f's{spread}_a{replicas}'
            configs[name]=(spread,replicas)
            phase1.extend(run(1,name,[base]*200,spread,replicas,range(10)))
    write_rows(root/'phase1.csv',phase1); write_rows(root/'phase1_per_rank.csv',detail)
    score=args.cache+'_proxy_ms'; traffic=args.cache+'_bytes'
    means=[]
    for name in ['naive_packed',*configs]:
        rows=[r for r in phase1 if r['policy']==name and r['split']=='train']
        means.append(dict(policy=name,**{k:float(np.mean([r[k] for r in rows])) for k in ['rho',score,traffic]}))
    frontier=pareto(means,['rho',traffic])
    eligible=[r for r in means if r['policy']!='naive_packed']
    ranked=sorted(eligible,key=lambda r:r[score])[:3]
    winner=ranked[0]['policy']; spread,replicas=configs[winner]
    baseline=next(r for r in means if r['policy']=='naive_packed')
    go=ranked[0][score] < baseline[score]
    decision=dict(selection_split='train prompts 0-7 only',cache=args.cache,assumed_tflops=args.tflops,
                  assumed_bandwidth_gbs=args.bandwidth_gbs,top3=ranked,pareto=frontier,go_phase2=go,
                  decision='placement' if go else 'prioritize communication overlap/routing',winner=winner)
    (root/'phase1_selection.json').write_text(json.dumps(decision,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,field in zip(axes,['rho',score]):
        for row in means:
            ax.scatter(row[traffic]/1e9,row[field]);ax.annotate(row['policy'],(row[traffic]/1e9,row[field]),fontsize=7)
        ax.set(xlabel='Train mean communication GB',ylabel=field)
    fig.tight_layout();fig.savefig(root/'phase1_pareto.png');plt.close(fig)
    # Later phases remain diagnostics when no-go; do not describe them as validated winners.
    prob=np.asarray(masks[:8],dtype=np.float32).mean(axis=0)  # step, layer, head, q, k
    static=prob.mean(axis=(0,1,2));sizes=layouts[0].sizes[layouts[0].prefix:]
    candidates={kind:video_layout(layouts[0],g,kind) for kind in ['contiguous','temporal','spatial','3d']}
    candidates.update({kind:graph_layout(static,sizes,g,base,kind) for kind in ['graph','greedy','swap']})
    phase2=[]
    for name,v in candidates.items():
        phase2.extend(run(2,name,[v]*200,spread,replicas,range(10),
                          lambda p,v=v:locality(masks[p],layouts[p],v,g)))
    write_rows(root/'phase2.csv',phase2)
    chosen=min(candidates,key=lambda name:np.mean([r[score] for r in phase2 if r['policy']==name and r['split']=='train']))
    v=candidates[chosen]
    predictions=[]
    for budget in [2,4,8]:
        pred=np.zeros_like(prob,dtype=bool)
        np.put_along_axis(pred,np.argsort(-prob,axis=-1,kind='stable')[...,:budget],True,axis=-1)
        np.savez_compressed(root/f'prefetch_top{budget}.npz', video_mask=pred)
        for p in [8,9]:
            metrics=prefetch_metrics(masks[p],pred,layouts[p].sizes[layouts[p].prefix:],v,g)
            metrics['predicted_remote_query_rank_fanout']=locality(pred,layouts[p],v,g)['remote_query_rank_fanout']
            metrics['overlap_upper_bound_ms']=metrics['useful_prefetch_bytes']/(args.bandwidth_gbs*1e9)*1000
            predictions.append(dict(prompt=p,budget=budget,**metrics))
    write_rows(root/'phase3_prefetch.csv',predictions)
    schedules={'fixed':[v]*200}
    perstep=[graph_layout(prob[s].mean(axis=(0,1)),sizes,g,v,'swap') for s in range(4)]
    schedules['step']=[perstep[i//50] for i in range(200)]
    ten=[];drift=[];previous=v;prior=static
    for i in range(200):
        s,l=divmod(i,50)
        current=prob[s,l].mean(0)
        if l%10==0:
            updated=graph_layout(prob[s,l:l+10].mean(axis=(0,1)),sizes,g,v,'swap')
        ten.append(updated)
        if np.abs(current-prior).mean()>.02:
            previous=graph_layout(current,sizes,g,previous,'swap');prior=current
        drift.append(previous)
    schedules['every10']=ten;schedules['drift']=drift
    adaptive=[]
    for name,maps in schedules.items():
        adaptive.extend(run(3,name,maps,spread,replicas,[8,9]))
    write_rows(root/'phase3_adaptation.csv',adaptive)
    hardware=[]
    costs={kind:topology_matrix(g,kind) for kind in ['nvlink','pcie_same_numa','pcie_cross_numa']}
    if args.topology_cost:
        cost=np.asarray(json.loads(args.topology_cost.read_text()),float)
        if cost.shape!=(g,g) or not np.isfinite(cost).all() or (cost<0).any() or np.any(np.diag(cost)):
            raise ValueError('Cost must be finite, nonnegative ranks x ranks with zero diagonal')
        costs['custom']=cost
    for kind,cost in costs.items():
        hw=graph_layout(static,sizes,g,v,'swap',cost=cost)
        (root/f'runtime_{kind}.json').write_text(json.dumps(dict(ranks=g,prefix_spread=spread,audio_replicas=replicas,video_owner=hw.tolist(),tile_sizes=sizes.tolist(),hardware_cost_seconds_per_byte=cost.tolist()),indent=2))
        for name,maps in [('static',[v]*200),('hardware',[hw]*200),*schedules.items()]:
            if name=='fixed':
                continue
            hardware.extend(run(3,kind+'_'+name,maps,spread,replicas,[8,9],cost=cost))
    write_rows(root/'phase3_hardware.csv',hardware)
    assignments=dict(ranks=g,prefix_spread=spread,audio_replicas=replicas,video_policy=chosen,
                     tile_sizes=sizes.tolist(),video_owner=v.tolist(),selection='train only',phase1_go=go,
                     schedules={name:[m.tolist() for m in maps] for name,maps in schedules.items()})
    (root/'runtime_layout.json').write_text(json.dumps(assignments,indent=2))
    files=[args.source/'video_masks.npy',*args.source.glob('layout_*.npz'),Path(__file__),Path(__file__).with_name('policies.py')]
    (root/'plan_provenance.json').write_text(json.dumps(dict(
        hashes={str(f):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in files},
        training=list(range(8)),heldout=[8,9],runtime_measured=False,
        drift_threshold=.02,selection=decision,hardware_costs={k:c.tolist() for k,c in costs.items()}),indent=2))
    print('Completed offline phases; Phase 4 requires benchmark_ring.py and real model inference.',flush=True)


if __name__=='__main__':
    main()
