# SPDX-License-Identifier: Apache-2.0
"""Fused inference accumulator for arbitrary H3 token ownership.

Each CTA owns query rows of one batch/head. Entire unselected key chunks are
skipped. FP32 online-softmax state persists across Ring steps, so a tile split
across owners and all-masked shards obey the same normalization as full attention.
"""
import triton
import triton.language as tl


@triton.jit
def _accumulate(Q, K, V, QT, KT, MASK, MAXIMUM, DENOM, NUMERATOR, NQ: tl.constexpr, NK: tl.constexpr,
                HEADS: tl.constexpr, DIM: tl.constexpr, TILES: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
                QS2: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr, VS0: tl.constexpr,
                VS1: tl.constexpr, VS2: tl.constexpr, SCALE: tl.constexpr, BD: tl.constexpr, BQ: tl.constexpr,
                BK: tl.constexpr):
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
    d = tl.arange(0, BD)
    qt = tl.load(QT + qi, qi < NQ, other=0)
    q = tl.load(Q + batch * QS0 + qi[:, None] * QS1 + head * QS2 + d[None, :], (qi[:, None] < NQ) & (d[None, :] < DIM),
                other=0)
    m = tl.load(MAXIMUM + bh * NQ + qi, qi < NQ, other=-float('inf'))
    denominator = tl.load(DENOM + bh * NQ + qi, qi < NQ, other=0.)
    acc = tl.load(NUMERATOR + (bh * NQ + qi[:, None]) * DIM + d[None, :], (qi[:, None] < NQ) & (d[None, :] < DIM),
                  other=0.)
    for start in range(tl.cdiv(NK, BK)):
        ki = start * BK + tl.arange(0, BK)
        kt = tl.load(KT + ki, ki < NK, other=0)
        valid = (qi[:, None] < NQ) & (ki[None, :] < NK)
        selected = tl.load(MASK + ((bh * TILES + qt[:, None]) * TILES + kt[None, :]), valid, other=0)
        if tl.sum(tl.sum(selected.to(tl.int32), 0), 0) > 0:
            kk = tl.load(K + batch * KS0 + ki[None, :] * KS1 + head * KS2 + d[:, None],
                         (ki[None, :] < NK) & (d[:, None] < DIM),
                         other=0)
            vv = tl.load(V + batch * VS0 + ki[:, None] * VS1 + head * VS2 + d[None, :],
                         (ki[:, None] < NK) & (d[None, :] < DIM),
                         other=0)
            scores = tl.dot(q, kk) * SCALE
            scores = tl.where(selected, scores, -float('inf'))
            new_m = tl.maximum(m, tl.max(scores, 1))
            shift = tl.where(new_m == -float('inf'), 0., new_m)
            alpha = tl.exp(m - shift)
            weights = tl.exp(scores - shift[:, None])
            acc = acc * alpha[:, None] + tl.dot(weights.to(vv.dtype), vv)
            denominator = denominator * alpha + tl.sum(weights, 1)
            m = new_m
    tl.store(MAXIMUM + bh * NQ + qi, m, qi < NQ)
    tl.store(DENOM + bh * NQ + qi, denominator, qi < NQ)
    tl.store(NUMERATOR + (bh * NQ + qi[:, None]) * DIM + d[None, :], acc, (qi[:, None] < NQ) & (d[None, :] < DIM))


def accumulate_cuda(q, k, v, qtiles, ktiles, mask, state, scale) -> None:
    if q.shape[1] == 0 or k.shape[1] == 0:
        return
    _accumulate[(triton.cdiv(q.shape[1], 32), q.shape[0] * q.shape[2])](q,
                                                                        k,
                                                                        v,
                                                                        qtiles,
                                                                        ktiles,
                                                                        mask.contiguous(),
                                                                        *state,
                                                                        q.shape[1],
                                                                        k.shape[1],
                                                                        q.shape[2],
                                                                        q.shape[3],
                                                                        mask.shape[-1],
                                                                        *q.stride()[:3],
                                                                        *k.stride()[:3],
                                                                        *v.stride()[:3],
                                                                        scale,
                                                                        max(16, triton.next_power_of_2(q.shape[-1])),
                                                                        32,
                                                                        64,
                                                                        num_warps=4)
