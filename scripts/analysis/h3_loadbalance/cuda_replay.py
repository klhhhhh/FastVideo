"""Single physical GPU, sequential virtual ranks, synthetic QKV + captured masks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from policies import Layout, owners, residency, evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/home/kun/output_h3_vsa/predictability-20261007'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required; do not report CPU replay as GPU validation'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261008)
    masks=np.load(args.source/'video_masks.npy',mmap_mode='r')
    assignments=json.loads((args.output/'assignments.json').read_text())
    rows=[]
    start=time.time()
    for prompt in [8,9]:
        layout=Layout.load(args.source/f'layout_{prompt:02d}.npz')
        token_tiles=torch.tensor(layout.tokens,device='cuda')
        q,k,v=[torch.randn(layout.n,128,device='cuda',dtype=torch.float32) for _ in range(3)]
        for head in [0,17]:
            observed=masks[prompt,2,20,head]
            tilemask=torch.ones(len(layout.sizes),len(layout.sizes),dtype=torch.bool,device='cuda')
            tilemask[layout.prefix:,layout.prefix:]=torch.tensor(observed,device='cuda')
            mask=tilemask[token_tiles[:,None],token_tiles[None,:]]
            ref=torch.softmax((q@k.T/128**.5).masked_fill(~mask,-torch.inf),dim=-1)@v
            for key,definition in assignments.items():
                if not key.startswith(f'p{prompt}_g8_'):
                    continue
                video=None if definition['video'] is None else np.array(definition['video'][20])
                own=owners(layout,8,video,definition['weighted'])
                resident=residency(layout,own,8,definition['text_replica'],definition['audio_groups'])
                estimated=evaluate(observed,layout,own,8,definition['text_replica'],definition['audio_groups'])
                replay=torch.empty_like(ref)
                demand=0
                for rank in range(8):
                    ids=torch.tensor(np.flatnonzero(own==rank),device='cuda')
                    selected=mask[ids]
                    needed=torch.nonzero(selected.any(0)).flatten()
                    demand+=int((~resident[needed.cpu().numpy(),rank]).sum())*512
                    # Materialize this virtual rank's required KV, keeping token identity.
                    local_k,local_v=k[needed].clone(),v[needed].clone()
                    result=torch.softmax((q[ids]@local_k.T/128**.5).masked_fill(~selected[:,needed],-torch.inf),dim=-1)@local_v
                    replay[ids]=result
                torch.testing.assert_close(replay,ref,rtol=2e-5,atol=2e-5)
                assert demand==int(estimated['union_demand'][0])
                rows.append(dict(policy=key,step=2,layer=20,head=head,max_abs_error=float((replay-ref).abs().max()),demand_bytes=demand))
                print(key,head,rows[-1]['max_abs_error'],flush=True)
    torch.cuda.synchronize()
    report=dict(device=torch.cuda.get_device_name(),physical_gpus_used=1,virtual_ranks=8,
                qkv='seeded synthetic FP32; original model activations unavailable',
                mask='captured H3 VSA mask; exact dense prefix plus sparse video mask',
                scope='attention output and union-demand validation only; no NCCL, model inference or speedup measurement',
                elapsed_validation_seconds=time.time()-start,checks=rows)
    (args.output/'cuda_validation.json').write_text(json.dumps(report,indent=2))
    print('PASS',len(rows),'checks',flush=True)


if __name__=='__main__':
    main()
