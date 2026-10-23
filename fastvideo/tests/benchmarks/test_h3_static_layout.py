# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the FastH3 static-layout experiments."""

from __future__ import annotations

import csv
import json

import pytest

from fastvideo.benchmarks.h3_static_layout.audio_kv_residency import run_audio_kv_residency_sweep
from fastvideo.benchmarks.h3_static_layout.joint_sweep import run_joint_static_layout_sweep
from fastvideo.benchmarks.h3_static_layout.model import (
    H3StaticLayoutPolicy,
    H3StaticLayoutShape,
    pareto_frontier,
    recommend_static_layouts,
    simulate_static_layout_policy,
)
from fastvideo.benchmarks.h3_static_layout.prefix_query_spread import run_prefix_query_spread_sweep
from fastvideo.benchmarks.h3_static_layout.reporting import write_static_layout_report


def _small_shape() -> H3StaticLayoutShape:
    return H3StaticLayoutShape(
        world_size=4,
        text_tokens=8,
        audio_tokens=16,
        video_tokens=64,
        video_selected_tokens_per_query=8,
        kv_bytes_per_token=128,
        link_bandwidth_bytes_per_ms=1024 * 1024,
        token_pair_work_per_ms=1_000_000,
    )


def test_static_layout_shape_rejects_single_rank_sweeps() -> None:
    shape = H3StaticLayoutShape(world_size=1)

    with pytest.raises(ValueError, match="world_size must be at least 2"):
        shape.validate()


def test_prefix_query_spread_sweep_reduces_rho_and_increases_communication() -> None:
    shape = _small_shape()
    results = run_prefix_query_spread_sweep(shape, prefix_spreads=(1, 2, 4), audio_replicas=1)

    assert [row.prefix_spread for row in results] == [1, 2, 4]
    assert results[0].load_balance_ratio > results[1].load_balance_ratio > results[2].load_balance_ratio
    assert results[2].load_balance_ratio == 1.0
    assert results[2].remote_video_bytes > results[0].remote_video_bytes
    assert results[2].total_communication_bytes > results[0].total_communication_bytes


def test_audio_kv_residency_sweep_trades_remote_demand_for_refresh_and_memory() -> None:
    shape = _small_shape()
    results = run_audio_kv_residency_sweep(shape, audio_replicas=(1, 2, 4), prefix_spread=1)
    baseline, replica_2, replica_4 = results

    assert [row.audio_replicas for row in results] == [1, 2, 4]
    assert baseline.remote_audio_bytes > replica_2.remote_audio_bytes > replica_4.remote_audio_bytes
    assert baseline.audio_replica_refresh_bytes < replica_2.audio_replica_refresh_bytes < replica_4.audio_replica_refresh_bytes
    assert replica_2.extra_audio_kv_memory_bytes == replica_2.audio_replica_refresh_bytes
    assert replica_4.remote_audio_bytes == 0
    assert replica_2.net_audio_replication_savings_bytes == (
        baseline.remote_audio_bytes - replica_2.remote_audio_bytes - replica_2.audio_replica_refresh_bytes)


def test_joint_static_layout_sweep_exposes_pareto_and_recommended_layouts() -> None:
    shape = _small_shape()
    results = run_joint_static_layout_sweep(shape, prefix_spreads=(1, 2, 4), audio_replicas=(1, 2, 4))

    assert len(results) == 9
    assert {(row.prefix_spread, row.audio_replicas) for row in results} == {
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 1),
        (4, 2),
        (4, 4),
    }

    comm_rho_frontier = pareto_frontier(results)
    comm_latency_frontier = pareto_frontier(results, y_metric="latency_proxy_ms")
    recommended = recommend_static_layouts(results, limit=2)

    assert comm_rho_frontier
    assert comm_latency_frontier
    assert len(recommended) == 2
    assert all(row in comm_rho_frontier for row in recommended)


def test_policy_simulation_remains_available_for_single_layout_checks() -> None:
    shape = _small_shape()
    narrow = simulate_static_layout_policy(shape, H3StaticLayoutPolicy(prefix_spread=1, audio_replicas=1))
    wide = simulate_static_layout_policy(shape, H3StaticLayoutPolicy(prefix_spread=4, audio_replicas=1),
                                         baseline_remote_audio_bytes=narrow.remote_audio_bytes)

    assert wide.load_balance_ratio < narrow.load_balance_ratio
    assert max(wide.token_pair_work_by_rank) == min(wide.token_pair_work_by_rank)


def test_static_layout_report_creates_json_csv_and_markdown(tmp_path) -> None:
    shape = _small_shape()
    results = run_joint_static_layout_sweep(shape, prefix_spreads=(1, 2), audio_replicas=(1, 2))

    paths = write_static_layout_report(
        results,
        tmp_path,
        shape=shape,
        experiment_name="H3 Joint Static Layout Sweep",
        top_k=2,
    )

    payload = json.loads(paths["json"].read_text())
    assert payload["experiment"] == "H3 Joint Static Layout Sweep"
    assert payload["shape"]["world_size"] == 4
    assert len(payload["results"]) == 4
    assert len(payload["recommended_static_layouts"]) == 2

    with paths["csv"].open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 4
    assert csv_rows[0]["prefix_spread"] == "1"

    markdown = paths["markdown"].read_text()
    assert "# H3 Joint Static Layout Sweep" in markdown
    assert "Communication vs Rho Pareto Frontier" in markdown
    assert "Full Sweep" in markdown
