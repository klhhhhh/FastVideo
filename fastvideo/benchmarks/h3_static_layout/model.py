# SPDX-License-Identifier: Apache-2.0
"""Shared cost model for FastH3 multimodal static-layout experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class H3StaticLayoutShape:
    """Token and cost assumptions for one H3 static-layout experiment."""

    world_size: int = 8
    text_tokens: int = 256
    audio_tokens: int = 4096
    video_tokens: int = 32768
    video_selected_tokens_per_query: int = 512
    kv_bytes_per_token: int = 2 * 56 * 128 * 2
    link_bandwidth_bytes_per_ms: float = 300 * 1024 * 1024 * 1024 / 1000
    token_pair_work_per_ms: float = 50_000_000

    def validate(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if min(self.text_tokens, self.audio_tokens, self.video_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if self.video_tokens <= 0:
            raise ValueError("video_tokens must be positive")
        if self.video_selected_tokens_per_query <= 0:
            raise ValueError("video_selected_tokens_per_query must be positive")
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")
        if self.link_bandwidth_bytes_per_ms <= 0:
            raise ValueError("link_bandwidth_bytes_per_ms must be positive")
        if self.token_pair_work_per_ms <= 0:
            raise ValueError("token_pair_work_per_ms must be positive")

    @property
    def prefix_tokens(self) -> int:
        return self.text_tokens + self.audio_tokens

    @property
    def total_kv_tokens(self) -> int:
        return self.text_tokens + self.audio_tokens + self.video_tokens


@dataclass(frozen=True)
class H3StaticLayoutPolicy:
    """One ``prefix_spread x audio_replicas`` static layout."""

    prefix_spread: int
    audio_replicas: int

    def validate(self, world_size: int) -> None:
        if self.prefix_spread <= 0 or self.prefix_spread > world_size:
            raise ValueError("prefix_spread must be in [1, world_size]")
        if self.audio_replicas <= 0 or self.audio_replicas > world_size:
            raise ValueError("audio_replicas must be in [1, world_size]")
        if world_size % self.prefix_spread != 0:
            raise ValueError("prefix_spread must divide world_size")
        if world_size % self.audio_replicas != 0:
            raise ValueError("audio_replicas must divide world_size")


@dataclass(frozen=True)
class H3StaticLayoutMetrics:
    """Computed metrics for one static-layout policy."""

    prefix_spread: int
    audio_replicas: int
    load_balance_ratio: float
    latency_proxy_ms: float
    token_pair_work_by_rank: tuple[int, ...]
    remote_kv_demand_bytes: int
    remote_text_bytes: int
    remote_audio_bytes: int
    remote_video_bytes: int
    audio_replica_refresh_bytes: int
    text_replica_refresh_bytes: int
    extra_audio_kv_memory_bytes: int
    max_send_bytes: int
    max_receive_bytes: int
    send_bytes_by_rank: tuple[int, ...]
    receive_bytes_by_rank: tuple[int, ...]
    destination_fanout_by_rank: tuple[int, ...]
    total_communication_bytes: int
    net_audio_replication_savings_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def divisors(world_size: int) -> tuple[int, ...]:
    """Return the positive divisors of ``world_size`` in ascending order."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return tuple(value for value in range(1, world_size + 1) if world_size % value == 0)


def run_static_layout_grid(
    shape: H3StaticLayoutShape,
    *,
    prefix_spreads: Iterable[int],
    audio_replicas: Iterable[int],
) -> list[H3StaticLayoutMetrics]:
    """Run a ``prefix_spread x audio_replicas`` grid and return sorted metrics."""
    shape.validate()
    baseline_remote_audio = simulate_static_layout_policy(
        shape,
        H3StaticLayoutPolicy(prefix_spread=1, audio_replicas=1),
        baseline_remote_audio_bytes=0,
    ).remote_audio_bytes

    results = []
    for prefix_spread in tuple(prefix_spreads):
        for audio_replica_count in tuple(audio_replicas):
            policy = H3StaticLayoutPolicy(prefix_spread=prefix_spread, audio_replicas=audio_replica_count)
            results.append(
                simulate_static_layout_policy(
                    shape,
                    policy,
                    baseline_remote_audio_bytes=baseline_remote_audio,
                ))
    return sorted(results, key=lambda item: (item.prefix_spread, item.audio_replicas))


def simulate_static_layout_policy(
    shape: H3StaticLayoutShape,
    policy: H3StaticLayoutPolicy,
    *,
    baseline_remote_audio_bytes: int | None = None,
) -> H3StaticLayoutMetrics:
    """Compute load-balance and communication metrics for one static layout."""
    shape.validate()
    policy.validate(shape.world_size)

    prefix_query_counts = _spread_evenly(shape.prefix_tokens, policy.prefix_spread, shape.world_size)
    video_query_counts = _spread_evenly(shape.video_tokens, shape.world_size, shape.world_size)

    work_by_rank = []
    for rank in range(shape.world_size):
        prefix_work = prefix_query_counts[rank] * shape.total_kv_tokens
        video_work = video_query_counts[rank] * (shape.text_tokens + shape.audio_tokens +
                                                 min(shape.video_selected_tokens_per_query, shape.video_tokens))
        work_by_rank.append(prefix_work + video_work)

    mean_work = sum(work_by_rank) / shape.world_size
    load_balance_ratio = max(work_by_rank) / mean_work if mean_work else 0.0

    send_bytes = [0 for _ in range(shape.world_size)]
    receive_bytes = [0 for _ in range(shape.world_size)]
    remote_text_bytes = 0
    remote_audio_bytes = 0
    remote_video_bytes = 0
    fanout: list[set[int]] = [set() for _ in range(shape.world_size)]

    for rank in range(shape.world_size):
        demands = _rank_kv_demands(shape, rank, has_prefix_queries=prefix_query_counts[rank] > 0)
        for text_token in demands.text:
            supplier = _remote_supplier(rank, tuple(range(shape.world_size)))
            if supplier is not None:
                bytes_ = shape.kv_bytes_per_token
                remote_text_bytes += bytes_
                _record_transfer(send_bytes, receive_bytes, fanout, supplier, rank, bytes_)
        for audio_token in demands.audio:
            supplier = _remote_supplier(rank, _audio_residents(shape, audio_token, policy.audio_replicas))
            if supplier is not None:
                bytes_ = shape.kv_bytes_per_token
                remote_audio_bytes += bytes_
                _record_transfer(send_bytes, receive_bytes, fanout, supplier, rank, bytes_)
        for video_token in demands.video:
            supplier = _remote_supplier(rank, (_contiguous_owner(video_token, shape.video_tokens, shape.world_size), ))
            if supplier is not None:
                bytes_ = shape.kv_bytes_per_token
                remote_video_bytes += bytes_
                _record_transfer(send_bytes, receive_bytes, fanout, supplier, rank, bytes_)

    audio_replica_refresh_bytes = shape.audio_tokens * max(0, policy.audio_replicas - 1) * shape.kv_bytes_per_token
    text_replica_refresh_bytes = shape.text_tokens * max(0, shape.world_size - 1) * shape.kv_bytes_per_token
    extra_audio_kv_memory_bytes = audio_replica_refresh_bytes
    remote_kv_demand_bytes = remote_text_bytes + remote_audio_bytes + remote_video_bytes
    total_communication_bytes = remote_kv_demand_bytes + audio_replica_refresh_bytes + text_replica_refresh_bytes
    if baseline_remote_audio_bytes is None:
        baseline_remote_audio_bytes = remote_audio_bytes
    net_audio_savings = baseline_remote_audio_bytes - remote_audio_bytes - audio_replica_refresh_bytes

    max_send = max(send_bytes) if send_bytes else 0
    max_receive = max(receive_bytes) if receive_bytes else 0
    compute_ms = max(work_by_rank) / shape.token_pair_work_per_ms
    communication_ms = max(max_send, max_receive) / shape.link_bandwidth_bytes_per_ms
    latency_proxy_ms = compute_ms + communication_ms

    return H3StaticLayoutMetrics(
        prefix_spread=policy.prefix_spread,
        audio_replicas=policy.audio_replicas,
        load_balance_ratio=load_balance_ratio,
        latency_proxy_ms=latency_proxy_ms,
        token_pair_work_by_rank=tuple(work_by_rank),
        remote_kv_demand_bytes=remote_kv_demand_bytes,
        remote_text_bytes=remote_text_bytes,
        remote_audio_bytes=remote_audio_bytes,
        remote_video_bytes=remote_video_bytes,
        audio_replica_refresh_bytes=audio_replica_refresh_bytes,
        text_replica_refresh_bytes=text_replica_refresh_bytes,
        extra_audio_kv_memory_bytes=extra_audio_kv_memory_bytes,
        max_send_bytes=max_send,
        max_receive_bytes=max_receive,
        send_bytes_by_rank=tuple(send_bytes),
        receive_bytes_by_rank=tuple(receive_bytes),
        destination_fanout_by_rank=tuple(len(peers) for peers in fanout),
        total_communication_bytes=total_communication_bytes,
        net_audio_replication_savings_bytes=net_audio_savings,
    )


def pareto_frontier(
    results: Iterable[H3StaticLayoutMetrics],
    *,
    x_metric: str = "total_communication_bytes",
    y_metric: str = "load_balance_ratio",
) -> list[H3StaticLayoutMetrics]:
    """Return non-dominated policies for two lower-is-better metrics."""
    items = list(results)
    frontier = []
    for candidate in items:
        x_value = getattr(candidate, x_metric)
        y_value = getattr(candidate, y_metric)
        dominated = False
        for other in items:
            if other is candidate:
                continue
            other_x = getattr(other, x_metric)
            other_y = getattr(other, y_metric)
            if other_x <= x_value and other_y <= y_value and (other_x < x_value or other_y < y_value):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (getattr(item, x_metric), getattr(item, y_metric)))


def recommend_static_layouts(results: Iterable[H3StaticLayoutMetrics],
                             *,
                             limit: int = 3) -> list[H3StaticLayoutMetrics]:
    """Pick a small set of candidate layouts from the communication/rho frontier."""
    frontier = pareto_frontier(results)
    if not frontier:
        return []

    min_comm = min(item.total_communication_bytes for item in frontier)
    max_comm = max(item.total_communication_bytes for item in frontier)
    min_rho = min(item.load_balance_ratio for item in frontier)
    max_rho = max(item.load_balance_ratio for item in frontier)

    def score(item: H3StaticLayoutMetrics) -> tuple[float, int, float]:
        comm_range = max(1, max_comm - min_comm)
        rho_range = max(1e-12, max_rho - min_rho)
        norm_comm = (item.total_communication_bytes - min_comm) / comm_range
        norm_rho = (item.load_balance_ratio - min_rho) / rho_range
        return (norm_comm + norm_rho, item.total_communication_bytes, item.load_balance_ratio)

    return sorted(frontier, key=score)[:limit]


@dataclass(frozen=True)
class _RankDemands:
    text: range
    audio: range
    video: range


def _rank_kv_demands(shape: H3StaticLayoutShape, rank: int, *, has_prefix_queries: bool) -> _RankDemands:
    text = range(shape.text_tokens)
    audio = range(shape.audio_tokens)
    video = range(shape.video_tokens) if has_prefix_queries else _selected_video_window(shape, rank)
    return _RankDemands(text=text, audio=audio, video=video)


def _selected_video_window(shape: H3StaticLayoutShape, rank: int) -> range:
    query_start, query_end = _contiguous_range(rank, shape.video_tokens, shape.world_size)
    selected = min(shape.video_selected_tokens_per_query, shape.video_tokens)
    radius = ceil(selected / 2)
    start = max(0, query_start - radius)
    end = min(shape.video_tokens, query_end + radius)
    return range(start, end)


def _spread_evenly(token_count: int, active_ranks: int, world_size: int) -> list[int]:
    counts = [0 for _ in range(world_size)]
    for rank in range(active_ranks):
        start, end = _contiguous_range(rank, token_count, active_ranks)
        counts[rank] = end - start
    return counts


def _contiguous_range(rank: int, token_count: int, parts: int) -> tuple[int, int]:
    start = token_count * rank // parts
    end = token_count * (rank + 1) // parts
    return start, end


def _contiguous_owner(token_index: int, token_count: int, world_size: int) -> int:
    return min(world_size - 1, token_index * world_size // token_count)


def _audio_residents(shape: H3StaticLayoutShape, token_index: int, replicas: int) -> tuple[int, ...]:
    owner = _contiguous_owner(token_index, shape.audio_tokens, shape.world_size)
    group_start = (owner // replicas) * replicas
    return tuple(range(group_start, group_start + replicas))


def _remote_supplier(rank: int, residents: tuple[int, ...]) -> int | None:
    if rank in residents:
        return None
    return min(residents, key=lambda resident: (abs(resident - rank), resident))


def _record_transfer(
    send_bytes: list[int],
    receive_bytes: list[int],
    fanout: list[set[int]],
    supplier: int,
    destination: int,
    bytes_: int,
) -> None:
    send_bytes[supplier] += bytes_
    receive_bytes[destination] += bytes_
    fanout[destination].add(supplier)
