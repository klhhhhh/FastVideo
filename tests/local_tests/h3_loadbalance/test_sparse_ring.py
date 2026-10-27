"""Real Gloo processes test transport, empty ranks, redistribution and sparse merge."""
import importlib.util
from pathlib import Path
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CORE = Path(__file__).resolve().parents[3] / 'fastvideo/attention/ring/h3_sparse.py'


def load_core():
    spec = importlib.util.spec_from_file_location('h3_sparse_core', CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(rank, world, address):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        core = load_core()
        torch.manual_seed(19)
        n, heads, dim = 13, 3, 8
        tiles = torch.tensor([0,2,1,3,2,1,3,2,4,4,3,4,4])
        sizes = torch.bincount(tiles)
        q,k,v = [torch.randn(2,n,heads,dim) for _ in range(3)]
        mask = torch.rand(2,heads,5,5) > .65
        mask[:,:,:1] = True
        mask[:,:,:,:1] = True
        # One head has no allowed keys on a tile: stable all-masked zero output.
        mask[:,0,4] = False
        allowed = mask[:,:,tiles[:,None],tiles[None,:]]
        scores = torch.einsum('bqhd,bkhd->bhqk',q.double(),k.double()) / dim**.5
        probs = torch.softmax(scores.masked_fill(~allowed,-torch.inf),-1).nan_to_num()
        ref = torch.einsum('bhqk,bkhd->bqhd',probs,v.double()).float()
        packed = torch.arange(n)//4
        for empty in [False,True]:
            owner = (torch.arange(n)*3)%world if not empty else torch.arange(n)%(world-1)
            ids = torch.where(owner==rank)[0]
            for transport in ['ring','demand']:
                resident = torch.nn.functional.one_hot(owner,world).bool()
                resident[tiles==0] = True
                resident[tiles==1, :2] = True
                stats={}
                with torch.no_grad():
                    out = core.sparse_ring_attention(q[:,ids],k[:,ids],v[:,ids],ids,owner,tiles,mask,
                                                      dist.group.WORLD,resident,transport,stats=stats)
                torch.testing.assert_close(out,ref[:,ids],atol=1e-6,rtol=2e-5)
                counts=torch.tensor([stats['kv_send_bytes'],stats['kv_recv_bytes']])
                dist.all_reduce(counts)
                assert counts[0]==counts[1]
            predicted=mask.clone()
            predicted[:,:,:,2:]=False
            pending=core.start_prefetch(k[:,ids],v[:,ids],ids,owner,tiles,predicted,dist.group.WORLD)
            out=core.sparse_ring_attention(q[:,ids],k[:,ids],v[:,ids],ids,owner,tiles,mask,
                                           dist.group.WORLD,resident,'demand',prefetched=pending)
            torch.testing.assert_close(out,ref[:,ids],atol=1e-6,rtol=2e-5)
            x=core.redistribute(q[:,torch.where(packed==rank)[0]],packed,owner,dist.group.WORLD)
            torch.testing.assert_close(x,q[:,ids],atol=0,rtol=0)
            back=core.redistribute(x,owner,packed,dist.group.WORLD)
            torch.testing.assert_close(back,q[:,torch.where(packed==rank)[0]],atol=0,rtol=0)
            pooled=core.pooled_tiles(q[:,ids],ids,tiles,sizes,dist.group.WORLD)
            expected=torch.stack([q[:,tiles==t].mean(1) for t in range(5)],dim=2)
            torch.testing.assert_close(pooled,expected,atol=1e-6,rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_real_four_process_sparse_ring(tmp_path):
    mp.spawn(worker,args=(4,'file://'+str(tmp_path/'rendezvous')),nprocs=4,join=True)


def test_reject_invalid_transport():
    with pytest.raises(ValueError,match='transport'):
        load_core().sparse_ring_attention(torch.zeros(1),torch.zeros(1),torch.zeros(1),None,None,None,None,None,transport='bad')


def integration_worker(rank, world, address):
    from types import SimpleNamespace
    import numpy as np
    import fastvideo.attention.h3_ring_attention as integration
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder, _build_block_mask
    from fastvideo.forward_context import set_forward_context
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=address,rank=rank,world_size=world,timeout=timedelta(seconds=90))
    try:
        integration.get_ring_size=lambda:world
        integration.get_sp_world_size=lambda:world
        integration.get_ring_group=lambda:SimpleNamespace(rank_in_group=rank,world_size=world,device_group=dist.group.WORLD)
        integration.get_current_fastvideo_args=lambda:SimpleNamespace(h3_ring_layout=None,h3_ring_transport='ring',
                                                                     h3_ring_schedule='fixed',h3_ring_prefetch=None,h3_ring_metrics=None,h3_ring_persistent=True)
        meta=MiniMaxH3VSAMetadataBuilder().build(current_timestep=1,patch_size=(1,1,1),VSA_sparsity=.75,
                                                packed_segments=(5,(4,8,8),6),device=torch.device('cpu'),tile_size=64)
        torch.manual_seed(2026)
        n=meta.total_seq_length;heads=3;dim=8;chunk=(n+world-1)//world
        q,k,v,gate=[torch.randn(1,n,heads,dim) for _ in range(4)]
        tiles=meta.untile_combined_index//64
        def pool(x):
            return torch.stack([x[:,tiles==t].mean(1) for t in range(len(meta.variable_block_sizes))],dim=2)
        pooled=pool(q) @ pool(k).transpose(-2,-1) / dim**.5
        selected=_build_block_mask(pooled,meta.num_prefix_tiles,.75,True,meta.video_tile_spans,meta.span_sparsities)
        scores=q.transpose(1,2) @ k.transpose(1,2).transpose(-2,-1)/dim**.5
        allowed=selected[:,:,tiles[:,None],tiles[None,:]]
        ref=(scores.masked_fill(~allowed,-torch.inf).softmax(-1) @ v.transpose(1,2)).transpose(1,2)
        ref=ref+gate*(pooled.softmax(-1) @ pool(v)).permute(0,2,1,3)[:,tiles]
        ids=torch.arange(rank*chunk,min((rank+1)*chunk,n))
        def shard(x):
            out=torch.full((1,chunk,heads,dim),100.)
            out[:,:len(ids)]=x[:,ids]
            return out
        layer=integration.H3RingAttention(heads,dim,prefix='blocks.0.attn').eval()
        for transport in ['ring','demand']:
            layer.transport=transport
            for fitted in [False,True]:
                layer.layout=None if not fitted else dict(ranks=world,video_owner=[0,1,2,3],prefix_spread=2,
                                                           audio_replicas=2,tile_sizes=[64]*4)
                for prefetch in ([False,True] if transport=='demand' else [False]):
                    layer.prior=np.zeros((2,1,heads,4,4),bool) if prefetch else None
                    if prefetch:
                        layer.prior[...,0]=True  # both misses and overfetch are corrected
                    with torch.no_grad(),set_forward_context(1,meta):
                        out,_=layer(shard(q),shard(k),shard(v),original_seq_len=n,gate_compress=shard(gate))
                    torch.testing.assert_close(out[:,:len(ids)],ref[:,ids],atol=2e-6,rtol=3e-5)
                    assert not out[:,len(ids):].any()
        # Persistent ownership runs attention and tokenwise work without a QKV
        # all-to-all per block, then restores the padded packed shard once.
        layer.prior=None
        session=integration.H3LayoutSession(n,torch.device('cpu'))
        with torch.no_grad(),set_forward_context(1,meta):
            hidden,newids=session.prepare(shard(q).flatten(2),layer)
            with session.activate(layer):
                result,_=layer(hidden.unflatten(-1,(heads,dim)),k[:,newids],v[:,newids],
                               original_seq_len=n,gate_compress=gate[:,newids])
            torch.testing.assert_close(result,ref[:,newids],atol=2e-6,rtol=3e-5)
            assert layer.last_stats['redistribution_send_bytes']==0
            layer.layout['video_owner']=[3,2,1,0]
            changed,changedids=session.prepare(result.flatten(2).square(),layer)
            torch.testing.assert_close(changed,ref[:,changedids].flatten(2).square(),atol=1e-5,rtol=5e-5)
            restored=session.restore(changed)
            torch.testing.assert_close(restored[:,:len(ids)],ref[:,ids].flatten(2).square(),atol=1e-5,rtol=5e-5)
            assert layer.active_owner is None
        dense=integration.H3DenseRingAttention(heads,dim).eval()
        with torch.no_grad():
            actual,_=dense(shard(q),shard(k),shard(v),original_seq_len=n)
        expected=(scores.softmax(-1) @ v.transpose(1,2)).transpose(1,2)
        torch.testing.assert_close(actual[:,:len(ids)],expected[:,ids],atol=2e-6,rtol=3e-5)
        # Run the actual H3 transformer driver, including text refiner, partial
        # RoPE, AdaLN, residuals, FFN and output heads with real Gloo collectives.
        import fastvideo.models.dits.minimax_h3 as h3
        from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
        from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSABackend
        h3.get_ring_size=lambda:world
        h3.get_sp_world_size=lambda:world
        h3.get_ulysses_size=lambda:1
        h3.model_parallel_is_initialized=lambda:True
        h3.get_sp_group=integration.get_ring_group
        h3.get_attn_backend=lambda *args,**kwargs:MiniMaxH3VSABackend
        def shard_sequence(x,dim):
            length=x.shape[dim];width=(length+world-1)//world
            shape=list(x.shape);shape[dim]=width*world
            padded=x.new_zeros(shape)
            padded.narrow(dim,0,length).copy_(x)
            return padded.narrow(dim,rank*width,width).contiguous(),length
        def gather_sequence(x,length,dim):
            chunks=[torch.empty_like(x) for _ in range(world)]
            dist.all_gather(chunks,x.contiguous())
            return torch.cat(chunks,dim).narrow(dim,0,length)
        h3.sequence_model_parallel_shard=shard_sequence
        h3.sequence_model_parallel_all_gather_with_unpad=gather_sequence
        arch=MiniMaxH3ArchConfig(num_attention_heads=3,attention_head_dim=8,rope_freq_dim=1,
                                hidden_size=24,ffn_dim=48,num_layers=2,num_refiner_layers=1,
                                in_channels=2,audio_in_channels=2,patch_size=(1,1,1),text_dim=8,
                                freq_dim=8,time_embed_hidden_dim=24,time_embed_dim=24)
        model=h3.MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch),{}).float().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=.08)
        inputs=dict(hidden_states=torch.randn(1,256,2),audio_hidden_states=torch.randn(1,6,2),
                    encoder_hidden_states=torch.randn(1,5,8),timestep=torch.tensor([.4]),
                    timestep_indices=torch.zeros(n,dtype=torch.long),token_tags=torch.arange(n)%3,
                    position_ids=torch.stack((torch.arange(n),torch.arange(n)%2,torch.arange(n)%3),-1),
                    text_indices=torch.arange(5),video_indices=torch.arange(5,261),audio_indices=torch.arange(261,267))
        with torch.no_grad(),set_forward_context(1,meta):
            baseline=model(**inputs)
            for index,block in enumerate(model.transformer_blocks):
                block.attn.distributed_attention.layout=dict(ranks=world,video_owner=[0,1,2,3] if index==0 else [3,2,1,0],
                                                             prefix_spread=2,audio_replicas=2,tile_sizes=[64]*4)
                block.attn.distributed_attention.transport='demand'
            balanced=model(**inputs)
        for a,b in zip(baseline,balanced):
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
    finally:
        dist.destroy_process_group()


def test_h3_selector_gate_layout_and_prefetch_integration(tmp_path):
    mp.spawn(integration_worker,args=(4,'file://'+str(tmp_path/'integration')),nprocs=4,join=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for fused accumulator')
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_fused_accumulator_cross_shard_and_empty_rows(dtype):
    core=load_core()
    torch.manual_seed(40)
    q,k,v=[torch.randn(2,37,3,128,device='cuda',dtype=dtype) for _ in range(3)]
    tiles=torch.arange(37,device='cuda')%7
    mask=torch.rand(2,3,7,7,device='cuda')>.6
    mask[:,:,0]=False
    allowed=mask[:,:,tiles[:,None],tiles[None,:]]
    scores=q.transpose(1,2).float() @ k.transpose(1,2).float().transpose(-2,-1)/128**.5
    ref=(scores.masked_fill(~allowed,-torch.inf).softmax(-1).nan_to_num() @ v.transpose(1,2).float()).transpose(1,2)
    shape=(2,3,37)
    state=(torch.full(shape,-torch.inf,device='cuda'),torch.zeros(shape,device='cuda'),
           torch.zeros((*shape,128),device='cuda'))
    for start,end in [(0,0),(7,19),(19,37),(0,7)]:
        core.accumulate(q,k[:,start:end],v[:,start:end],tiles,tiles[start:end],mask,state,128**-.5)
    out=(state[2]/state[1].clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).transpose(1,2)
    torch.testing.assert_close(out,ref,atol=.006,rtol=.03)
    assert torch.isfinite(out).all()
    assert not out[:,tiles==0].any()
