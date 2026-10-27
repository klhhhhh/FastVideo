"""Run on captured masks; no FastVideo runtime or distributed process is modified."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from policies import Layout, evaluate, owners, video_layout, refine, migration


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/home/kun/output_h3_vsa/predictability-20261007'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    masks = np.load(args.source / 'video_masks.npy', mmap_mode='r')
    assert masks.shape == (10, 4, 50, 56, 40, 40)
    assert np.all(masks.sum(-1) == 4)
    layouts = [Layout.load(args.source / f'layout_{p:02d}.npz') for p in range(10)]
    # No held-out prompt is used to fit or select swaps.
    prob = np.asarray(masks[:8], np.float32).mean(axis=(0, 1, 3))
    reference = layouts[0]
    summaries, latency, layers, assignments = [], [], [], {}
    start = time.time()
    for ranks in [2, 4, 8]:
        base = video_layout(reference, ranks, '3d')
        static = refine(prob.mean(0), reference, ranks, base)
        dynamic, minimal = [], []
        previous = static
        for layer in range(50):
            dynamic.append(refine(prob[layer], reference, ranks, base))
            previous = refine(prob[layer], reference, ranks, previous, previous, migration_weight=1.)
            minimal.append(previous)
        definitions = [('naive_packed', None, False, 1, False)]
        for geometry in ['temporal', 'spatial', '3d']:
            v = video_layout(reference, ranks, geometry)
            definitions.append((f'rr_{geometry}_shard', [v]*50, False, 1, False))
            definitions.append((f'rr_{geometry}_textrep', [v]*50, True, 1, False))
        definitions.append(('weighted_3d_textrep', [base]*50, True, 1, True))
        for groups in sorted({2, 4, ranks}):
            if groups <= ranks:
                definitions.append((f'audio_groups_{groups}', [base]*50, True, groups, False))
        definitions.extend([('predict_static', [static]*50, True, 1, False),
                            ('predict_layer', dynamic, True, 1, False),
                            ('predict_minimal', minimal, True, 1, False)])
        for prompt in [8, 9]:
            layout = layouts[prompt]
            initial = owners(layout, ranks)
            for name, maps, textrep, groups, weighted in definitions:
                key = f'p{prompt}_g{ranks}_{name}'
                owner_maps = [owners(layout, ranks, None if maps is None else maps[l], weighted) for l in range(50)]
                assignments[key] = {'video': None if maps is None else [v.tolist() for v in maps],
                                    'text_replica': textrep, 'audio_groups': groups, 'weighted': weighted}
                values = [evaluate(masks[prompt, :, l], layout, owner_maps[l], ranks, textrep, groups) for l in range(50)]
                # evaluate records are step-major then head-major. Aggregate heads
                # on each physical rank before applying the layer critical path.
                work = np.stack([v['work'].reshape(4, 56, ranks).sum(1) for v in values], axis=1).reshape(200, ranks)
                rho = work.max(1) / work.mean(1)
                moves = []
                prev = initial
                for step in range(4):
                    for l in range(50):
                        send, recv = migration(layout, prev, owner_maps[l], ranks)
                        moves.append(np.maximum(send, recv))
                        prev = owner_maps[l]
                moves = np.array(moves)
                migration_total = sum(migration(layout, owner_maps[(i-1) % 50], owner_maps[i % 50], ranks)[0].sum()
                                      for i in range(1, 200))
                row = dict(prompt=prompt, ranks=ranks, policy=name, rho_mean=float(rho.mean()),
                           rho_p95=float(np.quantile(rho, .95)), critical_pairs=int(work.max(1).sum()),
                           initial_hidden_bytes=int(migration(layout, initial, owner_maps[0], ranks)[0].sum()),
                           transition_hidden_bytes=int(migration_total),
                           extra_kv_peak_bytes=int(max(v['extra_bytes'].max() for v in values)*56),
                           resident_kv_peak_bytes=int(max(v['resident_bytes'].max() for v in values)*56))
                for mode in ['union', 'separate']:
                    recv = np.stack([v[mode+'_recv'].reshape(4, 56, ranks).sum(1) for v in values], axis=1).reshape(200, ranks)
                    send = np.stack([v[mode+'_send'].reshape(4, 56, ranks).sum(1) for v in values], axis=1).reshape(200, ranks)
                    refresh = sum(v['refresh_recv'].sum()*224 for v in values)
                    row[mode+'_demand_bytes'] = int(recv.sum()-refresh)
                    row['refresh_bytes'] = int(refresh)
                    row[mode+'_total_bytes'] = int(recv.sum())
                    if maps is not None:
                        assert recv.sum() == send.sum()
                        if mode == 'union':
                            assert recv.sum() == (ranks-1)*layout.n*512*56*200
                    traffic = np.maximum(send, recv)  # full-duplex logical link
                    for i in range(200):
                        for g in range(ranks):
                            layers.append(dict(prompt=prompt, ranks=ranks, policy=name, cache=mode,
                                               step=i//50, layer=i%50, rank=g, token_pairs=int(work[i,g]),
                                               send_bytes=int(send[i,g]), recv_bytes=int(recv[i,g]),
                                               migration_bottleneck_bytes=int(moves[i,g])))
                    for tflops in [10, 50, 200]:
                        for bandwidth in [10, 25, 100]:
                            for overlap in [0., .5, 1.]:
                                for sync_us in [0., 5.]:
                                    compute = work * (4*128) / (tflops*1e12)
                                    comm = traffic / (bandwidth*1e9)
                                    # Optimistic overlap continuum; no claim this schedule is achievable.
                                    body = (compute + comm - overlap*np.minimum(compute, comm)).max(1).sum()
                                    moving = np.any(moves > 0, axis=1)
                                    transfer = moves.max(1).sum() / (bandwidth*1e9)
                                    sync = (200 + moving.sum())*sync_us*1e-6
                                    for payload_multiplier in [1, 4]:
                                        latency.append(dict(prompt=prompt, ranks=ranks, policy=name, cache=mode,
                                                            tflops=tflops, bandwidth_GBs=bandwidth, overlap=overlap,
                                                            sync_us=sync_us, migration_payload_multiplier=payload_multiplier,
                                                            attention_ms=float(body*1000),
                                                            migration_ms=float(transfer*payload_multiplier*1000),
                                                            sync_ms=float(sync*1000),
                                                            proxy_ms=float((body+transfer*payload_multiplier+sync)*1000)))
                summaries.append(row)
                print(key, 'rho', round(row['rho_mean'],4), 'union GB', round(row['union_total_bytes']/1e9,4), flush=True)
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'latency_sweep.csv', latency)
    write_csv(out/'per_rank.csv', layers)
    (out/'assignments.json').write_text(json.dumps(assignments))
    files = [args.source/'video_masks.npy', *sorted(args.source.glob('layout_*.npz'))]
    hashes = {}
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        hashes[str(p)] = h.hexdigest()
    provenance = dict(source=str(args.source), input_sha256=hashes, training_prompts=list(range(8)),
                      heldout_prompts=[8,9], shape=list(masks.shape), execution='CPU offline logical ranks; not multi-GPU timing',
                      python=platform.python_version(), numpy=np.__version__, seconds=time.time()-start,
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                      source_provenance=json.loads((args.source/'provenance.json').read_text()))
    (out/'experiment.json').write_text(json.dumps(provenance, indent=2))
    print('DONE', len(summaries), 'configurations;', round(time.time()-start,2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
