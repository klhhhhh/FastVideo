# SPDX-License-Identifier: Apache-2.0
"""Audio KV residency sweep for FastH3 multimodal static layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo.benchmarks.h3_static_layout.cli import (
    add_shape_args,
    parse_int_list,
    print_written_paths,
    shape_from_args,
)
from fastvideo.benchmarks.h3_static_layout.model import (
    H3StaticLayoutMetrics,
    H3StaticLayoutShape,
    divisors,
    run_static_layout_grid,
)
from fastvideo.benchmarks.h3_static_layout.reporting import write_static_layout_report

EXPERIMENT_NAME = "H3 Audio KV Residency Sweep"


def run_audio_kv_residency_sweep(
    shape: H3StaticLayoutShape,
    *,
    audio_replicas: tuple[int, ...] | None = None,
    prefix_spread: int = 1,
) -> list[H3StaticLayoutMetrics]:
    """Sweep audio KV replica count while holding prefix query spread fixed."""
    replicas = audio_replicas if audio_replicas is not None else divisors(shape.world_size)
    return run_static_layout_grid(shape, prefix_spreads=(prefix_spread, ), audio_replicas=replicas)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FastH3 audio KV residency sweep.")
    add_shape_args(parser)
    parser.add_argument("--audio-replicas", default="1,2,4,8")
    parser.add_argument("--prefix-spread", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    shape = shape_from_args(args)
    results = run_audio_kv_residency_sweep(
        shape,
        audio_replicas=parse_int_list(args.audio_replicas, label="audio_replicas"),
        prefix_spread=args.prefix_spread,
    )
    output_dir = args.output_dir or Path("bench/h3_static_layout/audio_kv_residency")
    paths = write_static_layout_report(results,
                                       output_dir,
                                       shape=shape,
                                       experiment_name=EXPERIMENT_NAME,
                                       top_k=args.top_k)
    print_written_paths(paths)
    if args.print_markdown:
        print(paths["markdown"].read_text())


if __name__ == "__main__":
    main()
