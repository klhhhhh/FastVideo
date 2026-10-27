"""Run Phase 4 real H3 generation comparisons and preserve failures and telemetry.

Use compatible dense and VSA-trained checkpoints. Multi-node execution uses the
existing Ray launcher; run this harness on its head node. nvidia-smi samples
cover local GPUs only. Nsight traces are the source for idle/overlap analysis.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-model',required=True)
    p.add_argument('--vsa-model',required=True)
    p.add_argument('--layout',type=Path,required=True)
    p.add_argument('--prefetch',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-gpus',type=int,required=True)
    p.add_argument('--execution-backend',choices=['mp','ray'],default='mp')
    p.add_argument('--prompt',default='A red panda walking through a forest')
    p.add_argument('--height',type=int,default=256)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--num-frames',type=int,default=22)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--nsys',action='store_true')
    args=p.parse_args()
    definition=json.loads(args.layout.read_text())
    if args.num_gpus<2 or definition['ranks']!=args.num_gpus:
        p.error('Need >=2 GPUs and matching layout rank count')
    if args.nsys and shutil.which('nsys') is None:
        p.error('--nsys requires Nsight Systems installed')
    args.output.mkdir(parents=True,exist_ok=True)
    entry=Path(__file__).resolve().parents[3]/'examples/inference/basic/basic_minimax_h3_t2v.py'
    cases=[('h3_dense_sp',args.dense_model,'FLASH_ATTN',1,[]),
           ('h3_dense_ring',args.dense_model,'FLASH_ATTN',args.num_gpus,[]),
           ('h3_vsa_sp',args.vsa_model,'VIDEO_SPARSE_ATTN_H3',1,[]),
           ('h3_vsa_naive_ring',args.vsa_model,'VIDEO_SPARSE_ATTN_H3',args.num_gpus,[])]
    static=['--h3-ring-layout',str(args.layout.resolve())]
    cases += [('h3_vsa_static_ring',args.vsa_model,'VIDEO_SPARSE_ATTN_H3',args.num_gpus,static),
              ('h3_vsa_static_demand',args.vsa_model,'VIDEO_SPARSE_ATTN_H3',args.num_gpus,
               static+['--h3-ring-transport','demand'])]
    for schedule in ['step','every10','drift']:
        cases.append(('h3_vsa_'+schedule,args.vsa_model,'VIDEO_SPARSE_ATTN_H3',args.num_gpus,
                      static+['--h3-ring-transport','demand','--h3-ring-schedule',schedule]))
    if args.prefetch:
        cases.append(('h3_vsa_prefetch',args.vsa_model,'VIDEO_SPARSE_ATTN_H3',args.num_gpus,
                      static+['--h3-ring-transport','demand','--h3-ring-prefetch',str(args.prefetch.resolve())]))
    results=[]
    for name,model,backend,ring,extra in cases:
        out=args.output/name;out.mkdir(exist_ok=True)
        cmd=[sys.executable,str(entry),'--model-path',model,'--attention-backend',backend,
             '--num-gpus',str(args.num_gpus),'--ring-size',str(ring),'--execution-backend',args.execution_backend,
             '--prompt',args.prompt,'--height',str(args.height),'--width',str(args.width),
             '--num-frames',str(args.num_frames),'--steps',str(args.steps),'--seed',str(args.seed),
             '--repeats',str(args.repeats),'--output',str(out.resolve()),'--h3-ring-metrics',str((out/'payloads').resolve()),*extra]
        if args.nsys:
            cmd=['nsys','profile','--trace=cuda,nvtx,osrt','--force-overwrite=true','--output',str(out/'trace'),*cmd]
        monitor=None
        with (out/'gpu_utilization.csv').open('w') as telemetry, (out/'inference.log').open('w') as log:
            if shutil.which('nvidia-smi'):
                monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                          '--format=csv','--loop-ms=250'],stdout=telemetry,stderr=subprocess.DEVNULL)
            started=time.time()
            try:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            finally:
                if monitor:
                    monitor.terminate();monitor.wait()
        text=(out/'inference.log').read_text()
        times=[float(x) for x in re.findall(r'Generation time: ([\d.]+)s',text)]
        results.append(dict(case=name,command=cmd,returncode=result.returncode,wall_seconds=time.time()-started,
                            generation_seconds=times,steady_state_seconds=times[-1] if len(times)>1 else None,
                            timing_includes_payload_logging=True,rank_idle_time_ms=None,
                            rank_idle_source='Nsight trace when --nsys; not inferred from host wait',
                            output=str(out),model_inference_passed=result.returncode==0 and bool(times)))
        (args.output/'inference_comparison.json').write_text(json.dumps(results,indent=2))
        print(name,result.returncode,times,flush=True)
    if any(not r['model_inference_passed'] for r in results):
        raise SystemExit('Some real inference cases failed; see inference_comparison.json and logs')


if __name__=='__main__':
    main()
