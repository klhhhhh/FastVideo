# SPDX-License-Identifier: Apache-2.0
"""Report writers for FastH3 multimodal static-layout experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from fastvideo.benchmarks.h3_static_layout.model import (
    H3StaticLayoutMetrics,
    H3StaticLayoutShape,
    pareto_frontier,
    recommend_static_layouts,
)


def write_static_layout_report(
    results: Iterable[H3StaticLayoutMetrics],
    output_dir: Path,
    *,
    shape: H3StaticLayoutShape,
    experiment_name: str,
    top_k: int = 3,
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown artifacts for a static-layout experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    base_name = _safe_artifact_name(experiment_name)
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"
    markdown_path = output_dir / f"{base_name}.md"

    payload = {
        "experiment": experiment_name,
        "shape": asdict(shape),
        "results": [row.to_dict() for row in rows],
        "pareto_comm_vs_rho": [row.to_dict() for row in pareto_frontier(rows)],
        "pareto_comm_vs_latency": [row.to_dict() for row in pareto_frontier(rows, y_metric="latency_proxy_ms")],
        "recommended_static_layouts": [row.to_dict() for row in recommend_static_layouts(rows, limit=top_k)],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(rows, csv_path)
    markdown_path.write_text(
        format_static_layout_markdown(
            rows,
            shape=shape,
            experiment_name=experiment_name,
            top_k=top_k,
        ))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def format_static_layout_markdown(
    results: Iterable[H3StaticLayoutMetrics],
    *,
    shape: H3StaticLayoutShape,
    experiment_name: str,
    top_k: int = 3,
) -> str:
    """Render a compact Markdown summary for humans reviewing a sweep."""
    rows = list(results)
    recommended = recommend_static_layouts(rows, limit=top_k)
    comm_rho_frontier = pareto_frontier(rows)
    comm_latency_frontier = pareto_frontier(rows, y_metric="latency_proxy_ms")

    lines = [
        f"# {experiment_name}",
        "",
        "## Shape",
        "",
        "| field | value |",
        "| --- | ---: |",
    ]
    for key, value in asdict(shape).items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Recommended Static Layouts",
        "",
        "| rank | prefix_spread | audio_replicas | rho | total_comm_gib | latency_proxy_ms | net_audio_savings_gib |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for index, row in enumerate(recommended, start=1):
        lines.append("| {rank} | {prefix} | {audio} | {rho:.4f} | {comm:.4f} | {latency:.4f} | {savings:.4f} |".format(
            rank=index,
            prefix=row.prefix_spread,
            audio=row.audio_replicas,
            rho=row.load_balance_ratio,
            comm=_bytes_to_gib(row.total_communication_bytes),
            latency=row.latency_proxy_ms,
            savings=_bytes_to_gib(row.net_audio_replication_savings_bytes),
        ))

    lines.extend([
        "",
        "## Communication vs Rho Pareto Frontier",
        "",
        "| prefix_spread | audio_replicas | rho | total_comm_gib | remote_demand_gib | max_send_gib | max_recv_gib |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comm_rho_frontier:
        lines.append("| {prefix} | {audio} | {rho:.4f} | {comm:.4f} | {remote:.4f} | {send:.4f} | {recv:.4f} |".format(
            prefix=row.prefix_spread,
            audio=row.audio_replicas,
            rho=row.load_balance_ratio,
            comm=_bytes_to_gib(row.total_communication_bytes),
            remote=_bytes_to_gib(row.remote_kv_demand_bytes),
            send=_bytes_to_gib(row.max_send_bytes),
            recv=_bytes_to_gib(row.max_receive_bytes),
        ))

    lines.extend([
        "",
        "## Communication vs Latency Pareto Frontier",
        "",
        "| prefix_spread | audio_replicas | latency_proxy_ms | total_comm_gib | rho |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comm_latency_frontier:
        lines.append("| {prefix} | {audio} | {latency:.4f} | {comm:.4f} | {rho:.4f} |".format(
            prefix=row.prefix_spread,
            audio=row.audio_replicas,
            latency=row.latency_proxy_ms,
            comm=_bytes_to_gib(row.total_communication_bytes),
            rho=row.load_balance_ratio,
        ))

    lines.extend([
        "",
        "## Full Sweep",
        "",
        "| prefix_spread | audio_replicas | rho | remote_audio_gib | audio_refresh_gib | total_comm_gib | fanout_max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            "| {prefix} | {audio} | {rho:.4f} | {remote_audio:.4f} | {refresh:.4f} | {comm:.4f} | {fanout} |".format(
                prefix=row.prefix_spread,
                audio=row.audio_replicas,
                rho=row.load_balance_ratio,
                remote_audio=_bytes_to_gib(row.remote_audio_bytes),
                refresh=_bytes_to_gib(row.audio_replica_refresh_bytes),
                comm=_bytes_to_gib(row.total_communication_bytes),
                fanout=max(row.destination_fanout_by_rank) if row.destination_fanout_by_rank else 0,
            ))
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[H3StaticLayoutMetrics], csv_path: Path) -> None:
    scalar_fields = [
        "prefix_spread",
        "audio_replicas",
        "load_balance_ratio",
        "latency_proxy_ms",
        "remote_kv_demand_bytes",
        "remote_text_bytes",
        "remote_audio_bytes",
        "remote_video_bytes",
        "audio_replica_refresh_bytes",
        "text_replica_refresh_bytes",
        "extra_audio_kv_memory_bytes",
        "max_send_bytes",
        "max_receive_bytes",
        "total_communication_bytes",
        "net_audio_replication_savings_bytes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            row_dict = row.to_dict()
            writer.writerow({field: row_dict[field] for field in scalar_fields})


def _safe_artifact_name(value: str) -> str:
    name = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in name.split("_") if part) or "static_layout_sweep"


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)
