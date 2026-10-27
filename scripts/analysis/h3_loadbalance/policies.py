"""Offline token-exact H3 placement and logical unicast traffic model."""
from dataclasses import dataclass
import numpy as np

KV_BYTES = 2 * 128 * 2  # K and V, one head, BF16
HIDDEN_BYTES = 5376 * 2  # one BF16 hidden-state stream, excluding residuals


@dataclass
class Layout:
    sizes: np.ndarray
    tokens: np.ndarray
    coords: np.ndarray
    prefix: int

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d['sizes'].astype(int), d['packed_token_to_tile'].astype(int), d['coords'], int(d['prefix']))

    @property
    def n(self):
        return len(self.tokens)


def video_layout(layout, ranks, geometry):
    c = layout.coords.astype(int)
    if geometry == '3d':
        factors = {2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}[ranks]
        bins = np.minimum(c * factors // (c.max(0) + 1), np.array(factors) - 1)
        return np.ravel_multi_index(bins.T, factors)
    if geometry == 'contiguous':
        return np.arange(len(c)) * ranks // len(c)
    axes = (2, 1, 0) if geometry == 'temporal' else (0, 2, 1)
    order = np.lexsort(tuple(c[:, a] for a in axes))
    result = np.empty(len(c), int)
    result[order] = np.arange(len(c)) * ranks // len(c)
    return result


def owners(layout, ranks, video=None, weighted=False, prefix_spread=None):
    spread = ranks if prefix_spread is None else prefix_spread
    if not 1 <= spread <= ranks:
        raise ValueError("prefix_spread must be between 1 and ranks")
    if video is None:
        return np.minimum(np.arange(layout.n) // int(np.ceil(layout.n / ranks)), ranks - 1)
    out = np.empty(layout.n, int)
    prefix = layout.tokens < layout.prefix
    out[~prefix] = video[layout.tokens[~prefix] - layout.prefix]
    if not weighted:
        out[prefix] = np.arange(prefix.sum()) % spread
    else:
        vs = layout.sizes[layout.prefix:]
        cost = np.bincount(video, weights=vs, minlength=ranks) * (layout.sizes[:layout.prefix].sum() + .1 * vs.sum())
        for i in np.flatnonzero(prefix):
            g = int(cost[:spread].argmin())
            out[i] = g
            cost[g] += layout.n
    return out


def residency(layout, owner, ranks, text_replica, audio_groups):
    assert ranks % audio_groups == 0
    res = np.eye(ranks, dtype=bool)[owner]
    if text_replica:
        res[layout.tokens == 0] = True
    audio = (layout.tokens > 0) & (layout.tokens < layout.prefix)
    group_size = ranks // audio_groups
    for group in range(audio_groups):
        ids = np.flatnonzero(audio)
        res[ids, owner[ids] % group_size + group * group_size] = True
    return res


def evaluate(masks, layout, owner, ranks, text_replica=True, audio_groups=1):
    """One record per head. Receive and send include refresh; demand also separate."""
    masks = np.asarray(masks, bool).reshape(-1, len(layout.coords), len(layout.coords))
    q = np.zeros((len(layout.sizes), ranks), np.int64)
    np.add.at(q, (layout.tokens, owner), 1)
    res = residency(layout, owner, ranks, text_replica, audio_groups)
    # Routes[source, destination, tile]; prefer original owner as source. Since
    # it always retains KV this is valid, deterministic, and exposes send skew.
    routes = np.zeros((ranks, ranks, len(layout.sizes)), np.int64)
    refresh = np.zeros((ranks, ranks), np.int64)
    for dst in range(ranks):
        ids = np.flatnonzero(~res[:, dst])
        np.add.at(routes, (owner[ids], np.full(len(ids), dst), layout.tokens[ids]), 1)
        ids = np.flatnonzero(res[:, dst] & (owner != dst))
        np.add.at(refresh[:, dst], owner[ids], KV_BYTES)
    p = layout.prefix
    prefq = q[:p].sum(0)
    vq = q[p:]
    work = (masks.astype(np.int64) @ layout.sizes[p:]) @ vq
    work += prefq * layout.n + vq.sum(0) * layout.sizes[:p].sum()
    video_need = np.zeros((len(masks), ranks, len(q)), bool)
    video_need[:, :, :p] = (vq.sum(0) > 0)[None, :, None]
    video_need[:, :, p:] = np.einsum('bqk,qg->bgk', masks.astype(np.int64), vq > 0) > 0
    dense = np.broadcast_to((prefq > 0)[None, :, None], video_need.shape)
    result = {'work': work, 'extra_bytes': (res.sum(0) - np.bincount(owner, minlength=ranks)) * KV_BYTES,
              'resident_bytes': res.sum(0) * KV_BYTES, 'refresh_recv': refresh.sum(0), 'refresh_send': refresh.sum(1)}
    for mode, needs in [('union', (video_need | dense).astype(np.int64)),
                        ('separate', video_need.astype(np.int64) + dense.astype(np.int64))]:
        flows = np.einsum('bgk,sgk->bsg', needs, routes) * KV_BYTES
        result[mode + '_demand'] = flows.sum((1, 2))
        result[mode + '_flows'] = flows + refresh[None]
        for modality, sl in [('text', slice(0, 1)), ('audio', slice(1, p)), ('video', slice(p, None))]:
            result[mode + '_' + modality + '_demand'] = np.einsum(
                'bgk,sgk->b', needs[:, :, sl], routes[:, :, sl]) * KV_BYTES
        # Mean destinations per token, including explicit replica refresh.
        result[mode + '_destination_fanout'] = (np.einsum('bgk,sgk->b', needs > 0, routes) + refresh.sum() / KV_BYTES) / layout.n

        result[mode + '_recv'] = flows.sum(1) + refresh.sum(0)
        result[mode + '_send'] = flows.sum(2) + refresh.sum(1)
    assert np.all(work.sum(1) == (masks @ layout.sizes[p:] * layout.sizes[p:]).sum(1) +
                  layout.sizes[:p].sum() * (layout.n + layout.sizes[p:].sum()))
    return result


def migration(layout, old, new, ranks, bytes_per_token=HIDDEN_BYTES):
    changed = old != new
    return (np.bincount(old[changed], minlength=ranks) * bytes_per_token,
            np.bincount(new[changed], minlength=ranks) * bytes_per_token)


def refine(prob, layout, ranks, start, previous=None, migration_weight=0., swaps=2):
    """Bounded swaps; exact expected work, probability-based dedup demand estimate.

    Dedup probability uses independent query edges and is only a training proxy.
    Held-out evaluation always uses the original observed masks, never this proxy.
    """
    vs = layout.sizes[layout.prefix:]
    prefix = layout.sizes[:layout.prefix].sum()
    n = layout.n
    baseline = (ranks - 1) * n * KV_BYTES
    prefix_counts = np.bincount(np.arange(prefix) % ranks, minlength=ranks)
    def score(v):
        w = np.bincount(v, weights=vs * (prefix + prob @ vs), minlength=ranks) + prefix_counts * n
        # Prefix phase fetches all KV. Extra video phase with prefix text replicas.
        demand = 0.
        for g in range(ranks):
            need = 1 - np.prod(1 - prob[v == g], axis=0)
            demand += np.sum(need[v != g] * vs[v != g]) * KV_BYTES
        move = 0 if previous is None else vs[v != previous].sum() * HIDDEN_BYTES
        return w.max() / w.mean() + demand / baseline + migration_weight * move / baseline
    out = start.copy()
    for _ in range(swaps):
        current, best = score(out), None
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                if out[a] == out[b]:
                    continue
                trial = out.copy()
                trial[a], trial[b] = trial[b], trial[a]
                value = score(trial)
                if value < current - 1e-12:
                    current, best = value, trial
        if best is None:
            break
        out = best
    return out
