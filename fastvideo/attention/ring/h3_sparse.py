# SPDX-License-Identifier: Apache-2.0
"""Inference-only H3 sparse attention over real process groups.

The ring transport forwards canonical KV shards. The demand transport uses a
pairwise cyclic P2P schedule (direct sends, not multi-hop forwarding), refreshing
explicit replicas once and sending each demanded token once per destination.
CUDA BF16/FP16 uses a fused Triton accumulator that skips unselected key chunks.
CPU/FP32 uses the portable tiled oracle. Neither materializes full token scores.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def redistribute(x, old_owner, new_owner, group):
    """Uneven token all-to-all; x is in ascending global token order locally."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    old_ids = torch.where(old_owner == rank)[0]
    new_ids = torch.where(new_owner == rank)[0]
    send_ids = [torch.where(new_owner[old_ids] == g)[0] for g in range(world)]
    recv_ids = [torch.where(old_owner[new_ids] == g)[0] for g in range(world)]
    send_counts, recv_counts = [i.numel() for i in send_ids], [i.numel() for i in recv_ids]
    send = x[:, torch.cat(send_ids)].transpose(0, 1).contiguous()
    recv = torch.empty((new_ids.numel(), *send.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, send, recv_counts, send_counts, group=group)
    out = torch.empty_like(recv)
    out[torch.cat(recv_ids)] = recv
    return out.transpose(0, 1).contiguous()


def pooled_tiles(x, ids, token_tiles, sizes, group):
    pooled = torch.zeros((x.shape[0], len(sizes), *x.shape[2:]), device=x.device, dtype=torch.float32)
    pooled.index_add_(1, token_tiles[ids], x.float())
    dist.all_reduce(pooled, group=group)
    return (pooled / sizes[None, :, None, None]).permute(0, 2, 1, 3)


def accumulate(q, k, v, qtiles, ktiles, mask, state, scale, query_chunk=128, key_chunk=256) -> None:
    """Online softmax, including rows with zero selected keys in a shard."""
    if (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.shape[-1] <= 256
            and all(t.stride(-1) == 1 for t in (q, k, v))):
        from fastvideo.attention.ring.h3_sparse_triton import accumulate_cuda
        accumulate_cuda(q, k, v, qtiles, ktiles, mask, state, scale)
        return
    maximum, denominator, numerator = state
    for qi in range(0, q.shape[1], query_chunk):
        end = min(qi + query_chunk, q.shape[1])
        qq = q[:, qi:end].transpose(1, 2).float()
        for ki in range(0, k.shape[1], key_chunk):
            ke = min(ki + key_chunk, k.shape[1])
            allowed = mask[:, :, qtiles[qi:end, None], ktiles[None, ki:ke]]
            scores = torch.matmul(qq, k[:, ki:ke].transpose(1, 2).float().transpose(-2, -1)) * scale
            scores.masked_fill_(~allowed, -torch.inf)
            old = maximum[:, :, qi:end]
            new = torch.maximum(old, scores.amax(-1))
            # A finite shift avoids -inf - -inf; no keys gives zero weight.
            shift = torch.where(torch.isfinite(new), new, torch.zeros_like(new))
            alpha = torch.exp(old - shift)
            weights = torch.exp(scores - shift.unsqueeze(-1))
            numerator[:, :, qi:end] *= alpha.unsqueeze(-1)
            numerator[:, :, qi:end] += torch.matmul(weights, v[:, ki:ke].transpose(1, 2).float())
            denominator[:, :, qi:end] *= alpha
            denominator[:, :, qi:end] += weights.sum(-1)
            maximum[:, :, qi:end] = new


def start_prefetch(k, v, ids, owner, token_tiles, predicted_mask, group):
    """Post predicted remote KV before the current selector runs.

    Returns pending receives and a global token/destination presence table.
    Predictions never alter the attention mask; missing KV is fetched later.
    """
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    present = torch.zeros((len(owner), world), dtype=torch.bool, device=k.device)
    for dst in range(world):
        qtiles = torch.unique(token_tiles[owner == dst])
        needed = predicted_mask[:, :, qtiles].any(dim=(0, 1, 2))[token_tiles]
        present[:, dst] = needed & (owner != dst)
    ops, sends, receives = [], [], []
    sent = received = 0
    for other in range(world):
        if other == rank:
            continue
        peer = dist.get_global_rank(group, other) if group is not None else other
        send = torch.stack((k[:, present[ids, other]], v[:, present[ids, other]])).contiguous()
        kid = torch.where((owner == other) & present[:, rank])[0]
        recv = torch.empty((2, k.shape[0], len(kid), k.shape[2], k.shape[3]), device=k.device, dtype=k.dtype)
        sends.append(send)  # keep async send storage alive
        receives.append((kid, recv))
        if send.numel():
            ops.append(dist.P2POp(dist.isend, send, peer, group))
        if recv.numel():
            ops.append(dist.P2POp(dist.irecv, recv, peer, group))
        sent += send.numel() * send.element_size()
        received += recv.numel() * recv.element_size()
    requests = dist.batch_isend_irecv(ops) if ops else []
    return dict(present=present,
                sends=sends,
                receives=receives,
                requests=requests,
                send_bytes=sent,
                recv_bytes=received)


def sparse_ring_attention(q,
                          k,
                          v,
                          ids,
                          owner,
                          token_tiles,
                          mask,
                          group,
                          resident=None,
                          transport='ring',
                          scale=None,
                          stats=None,
                          prefetched=None):
    """BSHD tensors with arbitrary noncontiguous global token IDs.

    Every rank must use identical owner, mask, and residency metadata. Mask is
    [batch, heads, tiles, tiles]. No padding token participates in the softmax.
    Local tensors are ordered by global ID. Supports empty query/owner ranks.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError('H3 sparse Ring supports inference only')
    if transport not in ('ring', 'demand'):
        raise ValueError('transport must be ring or demand')
    if prefetched is not None and transport != 'demand':
        raise ValueError('Predictive prefetch requires demand transport')
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if q.shape != k.shape or k.shape != v.shape or q.shape[1] != ids.numel():
        raise ValueError('Expected equal Q/K/V with one row per local token ID')
    scale = q.shape[-1]**-.5 if scale is None else scale
    shape = (q.shape[0], q.shape[2], q.shape[1])
    state = (torch.full(shape, -torch.inf, device=q.device,
                        dtype=torch.float32), torch.zeros(shape, device=q.device, dtype=torch.float32),
             torch.zeros((*shape, q.shape[-1]), device=q.device, dtype=torch.float32))
    qtiles = token_tiles[ids]
    wire_send = wire_recv = refresh_bytes = 0

    def consume(kk, vv, kid, phase='remote') -> None:
        with torch.profiler.record_function('h3.' + phase + '_attention'):
            accumulate(q, kk, vv, qtiles, token_tiles[kid], mask, state, scale)

    def peer(g: int) -> int:
        return dist.get_global_rank(group, g) if group is not None else g

    if transport == 'ring':
        counts = torch.bincount(owner, minlength=world).tolist()
        capacity = max(counts)
        buf = torch.zeros((2, q.shape[0], capacity, q.shape[2], q.shape[3]), device=q.device, dtype=q.dtype)
        buf[0, :, :len(ids)], buf[1, :, :len(ids)] = k, v
        for step in range(world):
            reqs = []
            if step < world - 1:
                nxt = torch.empty_like(buf)
                reqs = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, buf, peer((rank + 1) % world), group),
                    dist.P2POp(dist.irecv, nxt, peer((rank - 1) % world), group)
                ])
                wire_send += buf.numel() * buf.element_size()
                wire_recv += buf.numel() * buf.element_size()
            source = (rank - step) % world
            kid = torch.where(owner == source)[0]
            consume(buf[0, :, :len(kid)], buf[1, :, :len(kid)], kid, phase='local' if step == 0 else 'remote')
            for req in reqs:
                req.wait()
            if reqs:
                buf = nxt
    else:
        if resident is None:
            resident = torch.nn.functional.one_hot(owner, world).bool()
        needed = []
        for dst in range(world):
            tiles = torch.unique(token_tiles[owner == dst])
            # A token is sent once even when several heads/queries need it.
            selected = mask[:, :, tiles].any(dim=(0, 1, 2))
            needed.append(selected[token_tiles])
        # Explicit replica refresh and on-demand fetch are disjoint. Pairwise
        # cyclic exchange works with subgroup ranks and uneven/empty messages.
        previous: torch.Tensor | None = None
        previous_ids: torch.Tensor | None = None
        for step in range(1, world + 1):
            reqs = []
            if step < world:
                dst, src = (rank + step) % world, (rank - step) % world
                send_keep = resident[ids, dst] | needed[dst][ids]
                if prefetched is not None:
                    send_keep &= ~prefetched['present'][ids, dst]
                source_ids = torch.where(owner == src)[0]
                recv_keep = resident[source_ids, rank] | needed[rank][source_ids]
                if prefetched is not None:
                    recv_keep &= ~prefetched['present'][source_ids, rank]
                kid = source_ids[recv_keep]
                outgoing = torch.stack((k[:, send_keep], v[:, send_keep])).contiguous()
                incoming = torch.empty((2, q.shape[0], len(kid), q.shape[2], q.shape[3]),
                                       device=q.device,
                                       dtype=q.dtype)
                if outgoing.numel():
                    reqs.append(dist.P2POp(dist.isend, outgoing, peer(dst), group))
                if incoming.numel():
                    reqs.append(dist.P2POp(dist.irecv, incoming, peer(src), group))
                requests = dist.batch_isend_irecv(reqs) if reqs else []
                wire_send += outgoing.numel() * outgoing.element_size()
                wire_recv += incoming.numel() * incoming.element_size()
                refresh_bytes += int(resident[ids,
                                              dst].sum()) * 2 * q.shape[0] * q.shape[2] * q.shape[3] * q.element_size()
            # First remote transfer is in flight while local attention runs.
            if step == 1:
                consume(k, v, ids, phase='local')
                if prefetched is not None:
                    for request in prefetched['requests']:
                        with torch.profiler.record_function('h3.p2p_wait'):
                            request.wait()
                    for prefetched_ids, buffer in prefetched['receives']:
                        consume(buffer[0], buffer[1], prefetched_ids)
                    wire_send += prefetched['send_bytes']
                    wire_recv += prefetched['recv_bytes']
            else:
                assert previous is not None and previous_ids is not None
                consume(previous[0], previous[1], previous_ids)
            if step < world:
                for request in requests:
                    with torch.profiler.record_function('h3.p2p_wait'):
                        request.wait()
                previous, previous_ids = incoming, kid
    if stats is not None:
        if prefetched is not None:
            actual_need = mask[:, :, torch.unique(qtiles)].any(dim=(0, 1, 2))[token_tiles]
            cached = prefetched['present'][:, rank]
            per_token = 2 * q.shape[0] * q.shape[2] * q.shape[3] * q.element_size()
            stats.update(prefetch_recv_bytes=prefetched['recv_bytes'],
                         useful_prefetch_bytes=int((cached & actual_need).sum()) * per_token,
                         wasted_prefetch_bytes=int((cached & ~actual_need).sum()) * per_token)
        stats.update(kv_send_bytes=wire_send,
                     kv_recv_bytes=wire_recv,
                     refresh_send_bytes=refresh_bytes,
                     transport=transport)
    return (state[2] / state[1].clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).transpose(1, 2).to(q.dtype)
