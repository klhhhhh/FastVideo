# SPDX-License-Identifier: Apache-2.0
"""H3 VSA delegate: pooled selector collectives and sparse KV transport.

Persistent model integration migrates hidden states only at layout boundaries
and restores packed ownership at the end of each transformer forward. Direct
attention calls also support QKV redistribution as a diagnostic fallback.
"""
import contextlib
import functools
import json

import numpy as np
from pathlib import Path

import torch
from torch import nn

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, _build_block_mask
from fastvideo.attention.ring.h3_sparse import pooled_tiles, redistribute, sparse_ring_attention, start_prefetch
from fastvideo.distributed.parallel_state import get_ring_group, get_ring_size, get_sp_world_size
from fastvideo.fastvideo_args import get_current_fastvideo_args
from fastvideo.forward_context import get_forward_context


@functools.lru_cache(maxsize=2)
def _load_prior(path):
    with np.load(path, allow_pickle=False) as data:
        return data['video_mask'].copy()


class H3RingAttention(nn.Module):

    def __init__(self, num_heads, head_size, causal=False, prefix='', **kwargs):
        super().__init__()
        if causal or get_ring_size() != get_sp_world_size():
            raise NotImplementedError('H3 VSA Ring currently requires noncausal pure Ring (ring_size=sp_size)')
        self.attn_impl = MiniMaxH3VSAImpl(num_heads, head_size, False, head_size**-.5, prefix=prefix)
        self.use_ring_attention = True
        self.layout = None
        self.active_owner = None
        self.persistent = True
        self.transport = 'ring'
        self.schedule = 'fixed'
        self.prior = None
        self.metrics = None
        try:
            args = get_current_fastvideo_args()
        except ValueError:
            args = None
        if args is not None:
            self.transport = args.h3_ring_transport
            self.metrics = args.h3_ring_metrics
            self.persistent = args.h3_ring_persistent
            self.schedule = args.h3_ring_schedule
            if args.h3_ring_prefetch:
                if self.transport != 'demand':
                    raise ValueError('H3 prefetch requires demand transport')
                self.prior = _load_prior(args.h3_ring_prefetch)
            if args.h3_ring_layout:
                self.layout = json.loads(Path(args.h3_ring_layout).read_text())
        self.last_stats = {}

    def placement(self, meta, packed_owner, world, step) -> tuple[torch.Tensor, torch.Tensor | None]:
        tiles = meta.untile_combined_index.to(packed_owner.device) // meta.tile_elems
        sizes = meta.variable_block_sizes.to(packed_owner.device)
        owner = packed_owner
        resident = None
        if self.layout is not None:
            definition = self.layout
            if definition['ranks'] != world:
                raise ValueError('Layout rank count differs from the Ring group')
            video = definition['video_owner']
            if self.schedule != 'fixed':
                index = step * 50 + self.attn_impl.layer_idx
                maps = definition['schedules'][self.schedule]
                if not 0 <= index < len(maps):
                    raise ValueError('Adaptive schedule does not cover this step/layer')
                video = maps[index]
            if len(video) != meta.num_video_tiles or not all(0 <= r < world for r in video):
                raise ValueError('Layout video geometry/rank IDs do not match this request')
            if definition.get('tile_sizes') is not None:
                expected = definition['tile_sizes']
                if sizes[meta.num_prefix_tiles:].tolist() != expected:
                    raise ValueError('Layout was fitted to a different video tile geometry')
            spread, replicas = definition['prefix_spread'], definition['audio_replicas']
            if not 1 <= spread <= world or replicas < 1 or world % replicas:
                raise ValueError('Invalid prefix spread or audio replica count')
            is_prefix = tiles < meta.num_prefix_tiles
            owner = torch.empty_like(packed_owner)
            owner[is_prefix] = torch.arange(int(is_prefix.sum()), device=packed_owner.device) % spread
            owner[~is_prefix] = torch.tensor(video,
                                             device=packed_owner.device)[tiles[~is_prefix] - meta.num_prefix_tiles]
            resident = torch.nn.functional.one_hot(owner, world).bool()
            # The fitted multimodal policy is T2V-only: text first, audio next,
            # one generated video. Ref2VA still works with naive Ring placement.
            segments = meta.packed_token_segments
            dense = [s for s in segments if isinstance(s, int)]
            if len(dense) != 2 or not isinstance(segments[0], int) or len(meta.video_tile_spans) != 1:
                raise ValueError('Fitted layout requires text/audio + one video; use naive Ring for Ref2VA')
            text_count = segments[0]
            resident[:text_count] = True
            audio = is_prefix.clone()
            audio[:text_count] = False
            audio_ids = torch.where(audio)[0]
            for replica in range(replicas):
                resident[audio_ids, owner[audio_ids] % (world // replicas) + replica * (world // replicas)] = True
        return owner, resident

    @torch.compiler.disable
    def forward(self, q, k, v, original_seq_len=None, freqs_cis=None, gate_compress=None):
        if self.training or (torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))):
            raise NotImplementedError('H3 VSA Ring is inference-only')
        if freqs_cis is not None:
            raise ValueError('H3 applies its partial RoPE before sparse Ring')
        context = get_forward_context()
        meta = context.attn_metadata
        group = get_ring_group()
        if group is None:
            raise RuntimeError('H3 Ring process group is not initialized')
        rank, world = group.rank_in_group, group.world_size
        pg = group.device_group
        n = meta.total_seq_length
        if original_seq_len != n:
            raise ValueError('Packed sequence and VSA metadata lengths differ')
        tiles = meta.untile_combined_index.to(q.device) // meta.tile_elems
        sizes = meta.variable_block_sizes.to(q.device)
        packed_owner = torch.arange(n, device=q.device) // ((n + world - 1) // world)
        input_owner = packed_owner if self.active_owner is None else self.active_owner
        ids = torch.where(input_owner == rank)[0]
        query, key, value = [x[:, :len(ids)].contiguous() for x in (q, k, v)]
        owner, resident = self.placement(meta, packed_owner, world, context.current_timestep)
        if self.active_owner is not None:
            if not torch.equal(owner, self.active_owner):
                raise ValueError('Persistent H3 layout and attention placement disagree')
        elif self.layout is not None:
            query, key, value = [redistribute(x, packed_owner, owner, pg) for x in (query, key, value)]
        local_ids = torch.where(owner == rank)[0]
        pending = None
        if self.prior is not None:
            step, layer = context.current_timestep, self.attn_impl.layer_idx
            if not (0 <= step < self.prior.shape[0] and 0 <= layer < self.prior.shape[1]):
                raise ValueError('Prefetch prior does not cover this step/layer')
            prior = torch.as_tensor(self.prior[step, layer], device=q.device)
            if prior.shape != (q.shape[2], meta.num_video_tiles, meta.num_video_tiles):
                raise ValueError('Prefetch prior geometry/heads differ from this request')
            prediction = torch.ones((q.shape[0], q.shape[2], len(sizes), len(sizes)), dtype=torch.bool, device=q.device)
            prediction[:, :, meta.num_prefix_tiles:, meta.num_prefix_tiles:] = prior
            pending = start_prefetch(key, value, local_ids, owner, tiles, prediction, pg)
        # Allreduce tile sums, not full QKV: split tiles and padding are exact.
        qp = pooled_tiles(query, local_ids, tiles, sizes, pg)
        kp = pooled_tiles(key, local_ids, tiles, sizes, pg)
        scores = qp @ kp.transpose(-2, -1) / q.shape[-1]**.5
        sparsity = 0. if self.attn_impl.layer_idx in meta.dense_layers else meta.VSA_sparsity
        mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                 meta.span_sparsities)
        output = sparse_ring_attention(query,
                                       key,
                                       value,
                                       local_ids,
                                       owner,
                                       tiles,
                                       mask,
                                       pg,
                                       resident=resident,
                                       transport=self.transport,
                                       stats=self.last_stats,
                                       prefetched=pending)
        if self.layout is not None and self.active_owner is None:
            output = redistribute(output, owner, packed_owner, pg)
        if gate_compress is not None:
            vp = pooled_tiles(v[:, :len(ids)], ids, tiles, sizes, pg)
            compressed = (torch.softmax(scores, dim=-1) @ vp).permute(0, 2, 1, 3).to(output.dtype)
            output = torch.addcmul(output, compressed[:, tiles[ids]], gate_compress[:, :len(ids)])
        per_token = q.shape[0] * q.shape[2] * q.shape[3] * q.element_size()
        moved_out = int(((input_owner == rank) & (owner != rank)).sum())
        moved_back = int(((owner == rank) & (input_owner != rank)).sum())
        self.last_stats.update(step=context.current_timestep,
                               layer=self.attn_impl.layer_idx,
                               rank=rank,
                               redistribution_send_bytes=(3 * moved_out + moved_back) * per_token,
                               selector_allreduce_payload_bytes=qp.numel() * 4 *
                               (3 if gate_compress is not None else 2))
        if self.metrics:
            directory = Path(self.metrics)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f'rank_{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(self.last_stats) + '\n')
        result = torch.zeros_like(q)
        result[:, :len(ids)] = output
        return result, None


class H3DenseRingAttention(nn.Module):
    """Dense text-refiner delegate for pure H3 Ring, without an FA2 dependency."""

    def __init__(self, num_heads, head_size, causal=False, **kwargs):
        super().__init__()
        if causal or get_ring_size() != get_sp_world_size():
            raise NotImplementedError('H3 dense refiner delegate requires noncausal pure Ring')
        self.attn_impl = None
        self.use_ring_attention = True

    @torch.compiler.disable
    def forward(self, q, k, v, original_seq_len=None, freqs_cis=None):
        if self.training:
            raise NotImplementedError('H3 Ring is inference-only')
        if freqs_cis is not None:
            raise ValueError('H3 handles RoPE before its Ring delegate')
        if original_seq_len is None or original_seq_len <= 0:
            raise ValueError('H3 Ring needs the positive unpadded sequence length')
        group = get_ring_group()
        if group is None:
            raise RuntimeError('H3 Ring process group is not initialized')
        owner = torch.arange(original_seq_len, device=q.device) // q.shape[1]
        ids = torch.where(owner == group.rank_in_group)[0]
        tiles = torch.zeros(original_seq_len, device=q.device, dtype=torch.long)
        mask = torch.ones((q.shape[0], q.shape[2], 1, 1), device=q.device, dtype=torch.bool)
        out = sparse_ring_attention(q[:, :len(ids)], k[:, :len(ids)], v[:, :len(ids)], ids, owner, tiles, mask,
                                    group.device_group)
        result = torch.zeros_like(q)
        result[:, :len(ids)] = out
        return result, None


class H3LayoutSession:
    """Persist hidden ownership through attention, residuals and tokenwise FFNs.

    The denoiser still consumes packed global outputs at each step boundary;
    restore once at the end of a transformer forward. Actual boundary transfers
    are logged separately from the offline persistent-across-step proxy.
    """

    def __init__(self, length, device):
        group = get_ring_group()
        self.group = group.device_group
        self.rank, self.world = group.rank_in_group, group.world_size
        self.chunk = (length + self.world - 1) // self.world
        self.packed_owner = torch.arange(length, device=device) // self.chunk
        self.owner = self.packed_owner
        self.metrics = None

    @classmethod
    def create(cls, blocks, length, device):
        delegates = [b.attn.distributed_attention for b in blocks]
        if not any(isinstance(d, H3RingAttention) and d.layout is not None and d.persistent for d in delegates):
            return None
        if not all(isinstance(d, H3RingAttention) and d.persistent for d in delegates):
            raise ValueError('Persistent H3 layout requires sparse Ring delegates on every main block')
        return cls(length, device)

    def _record(self, before, after, x, layer) -> None:
        if not self.metrics:
            return
        row = dict(kind='hidden_migration',
                   step=get_forward_context().current_timestep,
                   layer=layer,
                   rank=self.rank,
                   send_bytes=int(((before == self.rank) & (after != self.rank)).sum()) * x.shape[0] * x.shape[-1] *
                   x.element_size())
        directory = Path(self.metrics)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f'rank_{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    @torch.compiler.disable
    def prepare(self, hidden, delegate):
        context = get_forward_context()
        new, _ = delegate.placement(context.attn_metadata, self.packed_owner, self.world, context.current_timestep)
        old_ids = torch.where(self.owner == self.rank)[0]
        hidden = hidden[:, :len(old_ids)].contiguous()
        self.metrics = delegate.metrics
        if not torch.equal(self.owner, new):
            self._record(self.owner, new, hidden, delegate.attn_impl.layer_idx)
            hidden = redistribute(hidden, self.owner, new, self.group)
        self.owner = new
        return hidden, torch.where(new == self.rank)[0]

    @contextlib.contextmanager
    def activate(self, delegate):
        previous = delegate.active_owner
        delegate.active_owner = self.owner
        try:
            yield
        finally:
            delegate.active_owner = previous

    @torch.compiler.disable
    def restore(self, hidden):
        if not torch.equal(self.owner, self.packed_owner):
            self._record(self.owner, self.packed_owner, hidden, 'restore')
            hidden = redistribute(hidden, self.owner, self.packed_owner, self.group)
        self.owner = self.packed_owner
        result = hidden.new_zeros((hidden.shape[0], self.chunk, *hidden.shape[2:]))
        result[:, :hidden.shape[1]] = hidden
        return result
