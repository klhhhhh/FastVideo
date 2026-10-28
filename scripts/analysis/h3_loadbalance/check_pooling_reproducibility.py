"""Repeat CUDA tile pooling with deterministic algorithms off/on.

Uses fixed synthetic inputs and a captured layout. This diagnoses pooling only;
it does not establish full-model reproducibility or performance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    module_path = Path(__file__).resolve().parents[3] / 'fastvideo/attention/ring/h3_sparse.py'
    spec = importlib.util.spec_from_file_location('h3_pool', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        with np.load(args.layout) as layout:
            sizes = torch.tensor(layout['sizes'], device='cuda')
            tiles = torch.tensor(layout['packed_token_to_tile'], device='cuda')
        world = dist.get_world_size()
        owner = torch.arange(len(tiles), device='cuda') // ((len(tiles) + world - 1) // world)
        ids = torch.where(owner == rank)[0]
        torch.manual_seed(777)
        values = torch.randn(1, len(tiles), 56, 128, device='cuda')[:, ids].contiguous()
        rows = []
        for dtype in [torch.bfloat16, torch.float32]:
            value = values.to(dtype)
            for deterministic in [False, True]:
                torch.use_deterministic_algorithms(deterministic)
                reference = module.pooled_tiles(value, ids, tiles, sizes, dist.group.WORLD)
                changes, maximum = 0, 0.
                for _ in range(args.repeats):
                    actual = module.pooled_tiles(value, ids, tiles, sizes, dist.group.WORLD)
                    changes += int(not torch.equal(actual, reference))
                    maximum = max(maximum, float((actual - reference).abs().max()))
                rows.append(dict(rank=rank, dtype=str(dtype), deterministic=deterministic,
                                 changed_repeats=changes, repeats=args.repeats, max_abs_delta=maximum))
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(
                scope='Fixed synthetic inputs, captured layout, real pooled_tiles; not model parity or performance',
                results=gathered), indent=2) + '\n')
    finally:
        torch.use_deterministic_algorithms(previous)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
