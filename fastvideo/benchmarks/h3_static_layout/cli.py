# SPDX-License-Identifier: Apache-2.0
"""Shared CLI helpers for FastH3 static-layout experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo.benchmarks.h3_static_layout.model import H3StaticLayoutShape


def add_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--audio-tokens", type=int, default=4096)
    parser.add_argument("--video-tokens", type=int, default=32768)
    parser.add_argument("--video-selected-tokens-per-query", type=int, default=512)
    parser.add_argument("--kv-bytes-per-token", type=int, default=2 * 56 * 128 * 2)
    parser.add_argument("--link-bandwidth-gbps", type=float, default=300.0)
    parser.add_argument("--token-pair-work-per-ms", type=float, default=50_000_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-markdown", action="store_true", help="Print the Markdown report to stdout.")


def shape_from_args(args: argparse.Namespace) -> H3StaticLayoutShape:
    return H3StaticLayoutShape(
        world_size=args.world_size,
        text_tokens=args.text_tokens,
        audio_tokens=args.audio_tokens,
        video_tokens=args.video_tokens,
        video_selected_tokens_per_query=args.video_selected_tokens_per_query,
        kv_bytes_per_token=args.kv_bytes_per_token,
        link_bandwidth_bytes_per_ms=args.link_bandwidth_gbps * 1024 * 1024 * 1024 / 1000,
        token_pair_work_per_ms=args.token_pair_work_per_ms,
    )


def parse_int_list(raw: str, *, label: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must contain at least one integer")
    return values


def print_written_paths(paths: dict[str, Path]) -> None:
    labels = {"json": "JSON", "csv": "CSV", "markdown": "Markdown"}
    for key, label in labels.items():
        print(f"Wrote H3 static-layout {label}: {paths[key]}")
