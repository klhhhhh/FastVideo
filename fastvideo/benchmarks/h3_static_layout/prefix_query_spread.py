# SPDX-License-Identifier: Apache-2.0
"""Prefix query spread sweep for FastH3 multimodal static layouts."""

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

EXPERIMENT_NAME = "H3 Prefix Query Spread Sweep"


def run_prefix_query_spread_sweep(
    shape: H3StaticLayoutShape,
    *,
    prefix_spreads: tuple[int, ...] | None = None,
    audio_replicas: int = 1,
) -> list[H3StaticLayoutMetrics]:
    """Sweep prefix query spread while holding audio KV residency fixed."""
    spreads = prefix_spreads if prefix_spreads is not None else divisors(shape.world_size)
    return run_static_layout_grid(shape, prefix_spreads=spreads, audio_replicas=(audio_replicas, ))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FastH3 prefix query spread sweep.")
    add_shape_args(parser)
    parser.add_argument("--prefix-spreads", default="1,2,4,8")
    parser.add_argument("--audio-replicas", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    shape = shape_from_args(args)
    results = run_prefix_query_spread_sweep(
        shape,
        prefix_spreads=parse_int_list(args.prefix_spreads, label="prefix_spreads"),
        audio_replicas=args.audio_replicas,
    )
    output_dir = args.output_dir or Path("bench/h3_static_layout/prefix_query_spread")
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
