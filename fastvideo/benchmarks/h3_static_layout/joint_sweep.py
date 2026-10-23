# SPDX-License-Identifier: Apache-2.0
"""Joint prefix spread and audio KV residency sweep for FastH3 static layouts."""

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

EXPERIMENT_NAME = "H3 Joint Static Layout Sweep"


def run_joint_static_layout_sweep(
    shape: H3StaticLayoutShape,
    *,
    prefix_spreads: tuple[int, ...] | None = None,
    audio_replicas: tuple[int, ...] | None = None,
) -> list[H3StaticLayoutMetrics]:
    """Run the full ``prefix_spread x audio_replicas`` static-layout sweep."""
    spreads = prefix_spreads if prefix_spreads is not None else divisors(shape.world_size)
    replicas = audio_replicas if audio_replicas is not None else divisors(shape.world_size)
    return run_static_layout_grid(shape, prefix_spreads=spreads, audio_replicas=replicas)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FastH3 joint static-layout sweep.")
    add_shape_args(parser)
    parser.add_argument("--prefix-spreads", default="1,2,4,8")
    parser.add_argument("--audio-replicas", default="1,2,4,8")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    shape = shape_from_args(args)
    results = run_joint_static_layout_sweep(
        shape,
        prefix_spreads=parse_int_list(args.prefix_spreads, label="prefix_spreads"),
        audio_replicas=parse_int_list(args.audio_replicas, label="audio_replicas"),
    )
    output_dir = args.output_dir or Path("bench/h3_static_layout/joint_sweep")
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
