"""torchrun benchmark of real processes; synthetic QKV with captured H3 masks.

CUDA requires one distinct GPU per local rank. --device cpu is correctness-only
Gloo testing, never reported as GPU performance or end-to-end model inference.
"""
import argparse
import hashlib
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fastvideo.attention.ring.h3_sparse import redistribute, sparse_ring_attention, start_prefetch
from policies import Layout, owners, residency


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/kun/output_h3_vsa/predictability-20261007'))
    parser.add_argument('--layout',type=Path)
    parser.add_argument('--prefetch',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--prompt',type=int,default=8)
    parser.add_argument('--heads',type=int,default=2)
    parser.add_argument('--step',type=int,default=2)
    parser.add_argument('--layer',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if args.device=='cuda':
        if int(os.environ.get('LOCAL_WORLD_SIZE','1'))>torch.cuda.device_count():
            raise RuntimeError('CUDA benchmark requires a distinct physical GPU per local rank')
        torch.cuda.set_device(local_rank)
    device=torch.device(args.device,local_rank) if args.device=='cuda' else torch.device('cpu')
    torch.set_num_threads(1)
    dist.init_process_group('nccl' if args.device=='cuda' else 'gloo',timeout=timedelta(minutes=10))
    rank,world=dist.get_rank(),dist.get_world_size()
    try:
        layout=Layout.load(args.source/f'layout_{args.prompt:02d}.npz')
        observed=np.load(args.source/'video_masks.npy',mmap_mode='r')[args.prompt,args.step,args.layer,:args.heads]
        tiles=torch.tensor(layout.tokens,device=device)
        mask=torch.ones((1,args.heads,len(layout.sizes),len(layout.sizes)),dtype=torch.bool,device=device)
        mask[:,:,layout.prefix:,layout.prefix:]=torch.tensor(observed,device=device)
        torch.manual_seed(20261027)
        dtype=torch.bfloat16 if args.device=='cuda' else torch.float32
        q,k,v=[torch.randn(1,layout.n,args.heads,128,device=device,dtype=dtype) for _ in range(3)]
        old_np=owners(layout,world);old=torch.tensor(old_np,device=device)
        old_ids=torch.where(old==rank)[0]
        dense=torch.ones_like(mask)
        policies=[('dense_ring',old_np,False,1,'ring',dense),('naive_vsa_ring',old_np,False,1,'ring',mask),
                  ('naive_vsa_demand',old_np,False,1,'demand',mask)]
        if args.layout:
            definition=json.loads(args.layout.read_text())
            if definition['ranks']!=world:
                raise ValueError('Layout ranks must match torchrun world size')
            new=owners(layout,world,np.array(definition['video_owner']),prefix_spread=definition['prefix_spread'])
            for transport in ['ring','demand']:
                policies.append(('static_'+transport,new,True,definition['audio_replicas'],transport,mask))
        if args.prefetch:
            predicted=mask.clone()
            prior=np.load(args.prefetch)['video_mask'][args.step,args.layer,:args.heads]
            predicted[:,:,layout.prefix:,layout.prefix:]=torch.tensor(prior,device=device)
            last=policies[-1]
            policies.append(('prefetch_demand',last[1],last[2],last[3],'demand',mask))
        else:
            predicted=None
        if args.layout:
            for schedule in ['step','every10','drift']:
                mapping=definition['schedules'][schedule][args.step*50+args.layer]
                own=owners(layout,world,np.array(mapping),prefix_spread=definition['prefix_spread'])
                policies.append((schedule+'_demand',own,True,definition['audio_replicas'],'demand',mask))
        rows=[]
        def sync():
            if device.type=='cuda':
                torch.cuda.synchronize()
        with torch.no_grad():
            for name,own,rep,groups,transport,selected in policies:
                owner=torch.tensor(own,device=device);ids=torch.where(owner==rank)[0]
                resident=torch.tensor(residency(layout,own,world,rep,groups),device=device)
                local=[x[:,old_ids].contiguous() for x in (q,k,v)]
                def execute():
                    moved=[redistribute(x,old,owner,dist.group.WORLD) for x in local] if not np.array_equal(own,old_np) else local
                    stats={}
                    pending=start_prefetch(moved[1],moved[2],ids,owner,tiles,predicted,dist.group.WORLD) if name=='prefetch_demand' else None
                    out=sparse_ring_attention(*moved,ids,owner,tiles,selected,dist.group.WORLD,resident,transport,stats=stats,prefetched=pending)
                    if not np.array_equal(own,old_np):
                        out=redistribute(out,owner,old,dist.group.WORLD)
                    return out,stats
                for _ in range(args.warmup):
                    execute()
                times=[]
                for _ in range(args.repeats):
                    dist.barrier();sync();start=time.perf_counter()
                    out,stats=execute();sync()
                    times.append((time.perf_counter()-start)*1000)
                # Dense independent oracle, query-chunked to bound reference memory.
                error=0.
                for start in range(0,len(old_ids),128):
                    qi=old_ids[start:start+128]
                    allowed=selected[:,:,tiles[qi,None],tiles[None,:]]
                    scores=q[:,qi].transpose(1,2).float() @ k.transpose(1,2).float().transpose(-2,-1)/128**.5
                    ref=(scores.masked_fill(~allowed,-torch.inf).softmax(-1) @ v.transpose(1,2).float()).transpose(1,2).to(dtype)
                    torch.testing.assert_close(out[:,start:start+len(qi)],ref,atol=.004 if dtype==torch.bfloat16 else 2e-6,rtol=.02)
                    error=max(error,float((out[:,start:start+len(qi)]-ref).abs().max()))
                stats['effective_kv_send_GBs']=stats['kv_send_bytes']/(np.median(times)*1e6)
                stats['redistribution_send_bytes']=int((old_np!=own)[old_np==rank].sum()*3+(old_np!=own)[own==rank].sum())*args.heads*128*q.element_size()
                local_row=dict(policy=name,rank=rank,median_ms=float(np.median(times)),max_abs_error=error,
                               measured_samples_ms=times,**stats)
                gathered=[None]*world
                dist.all_gather_object(gathered,local_row)
                if rank==0:
                    rows.extend(gathered)
                    print(name,max(r['median_ms'] for r in gathered),'ms',flush=True)
                if args.profile:
                    activities=[torch.profiler.ProfilerActivity.CPU]
                    if device.type=='cuda':
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as prof:
                        execute();sync()
                    args.output.mkdir(parents=True,exist_ok=True)
                    prof.export_chrome_trace(str(args.output/f'{name}_rank{rank}.json'))
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'distributed_attention.json').write_text(json.dumps(dict(
                torch_version=torch.__version__,code_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).resolve().parents[3]/'fastvideo/attention/ring/h3_sparse.py',Path(__file__).resolve().parents[3]/'fastvideo/attention/ring/h3_sparse_triton.py']},world_size=world,device=str(device),hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                scope='attention microbenchmark; captured mask and synthetic QKV; includes per-call QKV redistribution',
                gpu_performance_measured=device.type=='cuda',model_inference=False,results=rows),indent=2))
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
