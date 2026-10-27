"""Cross-check all ten prompts against independently archived trace analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from policies import Layout, evaluate, owners

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=Path('/home/kun/output_h3_vsa/predictability-20261007'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
m=np.load(args.source/'video_masks.npy',mmap_mode='r')
origin=Path(json.loads((args.source/'provenance.json').read_text())['source'])
rows=[]
for prompt in range(10):
    layout=Layout.load(args.source/f'layout_{prompt:02d}.npz')
    archived=json.loads((origin/f'prompt_{prompt:02d}/analysis/summary.json').read_text())
    for ranks in [2,4,8]:
        value=evaluate(m[prompt],layout,owners(layout,ranks),ranks,False,1)
        reference=archived['comparisons'][str(ranks)]['naive']
        np.testing.assert_array_equal(value['work'].sum(0),reference['token_pairs_g'])
        assert int(value['union_demand'].sum())==reference['demand_bytes']
        np.testing.assert_array_equal(value['union_send'].sum(0),reference['send_bytes_g'])
        np.testing.assert_array_equal(value['union_recv'].sum(0),reference['receive_bytes_g'])
        rows.append(dict(prompt=prompt,ranks=ranks,token_pairs=True,demand=True,send=True,receive=True))
        print('PASS',prompt,ranks,flush=True)
(args.output/'baseline_validation.json').write_text(json.dumps(rows,indent=2))
