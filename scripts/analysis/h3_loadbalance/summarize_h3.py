"""Summarize matched fixed-layout H3 runs; never substitute proxy latency."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
import torch


def summarize(root):
    rows, details = [], {}
    reference = None
    signature = None
    reference_schedule = None
    for name in ['A','B','C']:
        path = root/name
        config = json.loads((path/'run_config.json').read_text())
        current = {k:config[k] for k in ['model','seed','height','width','num_frames','sigma_grid_points',
                                        'sparsity','tile_size','prompts','start','count','repeats']}
        if signature is None:
            signature = current
        if current != signature:
            raise ValueError('Unmatched generation settings')
        ranks = [[json.loads(line) for line in (path/f'measurements_rank_{rank}.jsonl').read_text().splitlines()]
                 for rank in range(2)]
        if any(len(records) != 4 for records in ranks):
            raise ValueError('Expected exactly four repeats on each rank')
        for records in ranks:
            for record in records:
                actual = {k:record[k] for k in ['video_sigmas','audio_sigmas','video_timesteps','audio_timesteps']}
                if reference_schedule is None:
                    reference_schedule = actual
                if actual != reference_schedule:
                    raise ValueError('Actual modality schedules differ')
        if len(list(path.glob('latents_*.pt'))) != 4 or len(list(path.glob('video_*.mp4'))) != 4:
            raise ValueError('Missing preserved latent or video output')
        results = [json.loads(line.split('GENERATION_RESULT ',1)[1]) for line in (path/'run.log').read_text().splitlines()
                   if line.startswith('GENERATION_RESULT ')]
        if len(results) != 4:
            raise ValueError('Missing successful generation results')
        times = [r['generation_seconds'] for r in results]
        stages = [max(ranks[r][i]['denoising_stage_seconds'] for r in range(2)) for i in range(4)]
        cuda = [max(sum(ranks[r][i]['transformer_cuda_ms']) for r in range(2))/1000 for i in range(4)]
        peak = max(r['peak_allocated_mib'] for records in ranks for r in records)
        comparisons = []
        decoded = []
        for latent in sorted(path.glob('latents_*.pt')):
            value = torch.load(latent,weights_only=True)
            if reference is None:
                reference = value
            compare = dict(file=latent.name)
            for modality,tensor in value.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Nonfinite output: {latent}')
                delta = tensor.float()-reference[modality].float()
                compare[modality] = dict(equal=torch.equal(tensor,reference[modality]),
                    max_abs=float(delta.abs().max()),rmse=float(delta.square().mean().sqrt()),
                    reference_rms=float(reference[modality].float().square().mean().sqrt()))
            comparisons.append(compare)
        for video in sorted(path.glob('video_*.mp4')):
            item = dict(file=video.name)
            for modality,stream,dtype,fmt in [('video','v',np.uint8,'rawvideo'),('audio','a',np.float32,'f32le')]:
                command = ['ffmpeg','-v','error','-i',str(video),'-map',f'0:{stream}:0']
                command += ['-pix_fmt','rgb24'] if modality == 'video' else ['-acodec','pcm_f32le']
                command += ['-f',fmt,'-']
                data = np.frombuffer(subprocess.check_output(command),dtype=dtype).astype(np.float32)
                key = '_reference_'+modality
                if key not in details:
                    details[key] = data
                baseline = details[key]
                if data.shape != baseline.shape:
                    raise ValueError('Decoded output geometry differs')
                delta = data-baseline
                item[modality] = dict(max_abs=float(np.abs(delta).max()),mae=float(np.abs(delta).mean()),
                                      rmse=float(np.sqrt(np.mean(delta**2))))
            decoded.append(item)
        rows.append(dict(case=name,first_generation_s=times[0],later_median_s=statistics.median(times[1:]),
            later_min_s=min(times[1:]),later_max_s=max(times[1:]),later_stdev_s=statistics.pstdev(times[1:]),
            denoising_stage_later_median_s=statistics.median(stages[1:]),
            transformer_cuda_span_later_median_s=statistics.median(cuda[1:]),peak_allocated_mib=peak))
        details[name] = dict(generation_seconds=times,stage_seconds=stages,transformer_cuda_span_seconds=cuda,
                             latent_comparison_to_A_first=comparisons,decoded_comparison_to_A_first=decoded)
    details = {k:v for k,v in details.items() if not k.startswith('_reference_')}
    details['actual_schedule'] = reference_schedule
    details['scope'] = ('Cached real text conditioning; lazy FSDP DiT and decoder reload; payload/profiler disabled. '
                        'Stage includes loading; CUDA forward spans include communication waits. '
                        'Peak memory is worker lifetime high-water mark. Four repeats, later statistics use three samples.')
    with (root/'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/'comparison.json').write_text(json.dumps(dict(rows=rows,details=details),indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root)
