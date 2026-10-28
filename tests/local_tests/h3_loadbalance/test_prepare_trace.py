"""Reject corrupt/incomplete captures before publishing offline trace files."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('prepare_h3_trace', Path(__file__).resolve().parents[3] /
                                              'scripts/analysis/h3_loadbalance/prepare_trace.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'raw'
    trace = root/'prompt_00/trace'
    layout_dir = trace/'layouts'
    records = trace/'samples/sample_0/rank_0'
    layout_dir.mkdir(parents=True)
    records.mkdir(parents=True)
    sizes = np.ones(42, dtype=np.int64)
    positions = np.stack([np.arange(42),np.zeros(42),np.zeros(42)],axis=1).astype(np.int64)
    torch.save(dict(layout_id='test',variable_block_sizes=torch.from_numpy(sizes),
                    packed_token_to_tile=torch.arange(42),position_ids=torch.from_numpy(positions),
                    tile_modalities=['text','audio']+['video']*40,num_prefix_tiles=2,tile_shape=[4,4,4]),
               layout_dir/'layout.pt')
    mask = np.zeros((1,2,42,42), dtype=bool)
    mask[:,:,:2,:] = True
    mask[:,:,:,:2] = True
    mask[:,:,2:,2:6] = True
    for step in range(2):
        for layer in range(2):
            torch.save(dict(encoding='packed_bool',bitorder='little',layout_id='test',
                global_head_ids=[0,1],head_dim=128,element_size=2,vsa_mode='exempt',
                selection_is_dense=False,dense_override=False,call_id=0,step_id=step,forward_id=step,
                layer_id=layer,mask_shape=list(mask.shape),
                mask_bits=torch.from_numpy(np.packbits(mask.reshape(-1),bitorder='little'))),
                records/f'{step}_{layer}.pt')
    (records.parent/'sample.json').write_text(json.dumps(dict(complete=True,model='fixture',seed=1000,
        video_sigmas=[1,.5,0],audio_sigmas=[1,.5,0],video_timesteps=[0,.5],audio_timesteps=[0,.5])))
    return root


def test_round_trip_and_complete_publication(source, tmp_path):
    out = tmp_path/'export'
    prepare.convert(source,out,prompts=1,steps=2,layers=2,heads=2)
    masks = np.load(out/'video_masks.npy')
    assert masks.shape == (1,2,2,2,40,40)
    assert masks.dtype == np.bool_
    assert masks[..., :4].all()
    assert not masks[..., 4:].any()
    assert json.loads((out/'provenance.json').read_text())['complete']
    assert not out.with_name('export.incomplete').exists()


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'heads', 'prefix', 'rank'])
def test_corruption_never_publishes(source, tmp_path, corruption):
    records = source/'prompt_00/trace/samples/sample_0/rank_0'
    path = records/'0_0.pt'
    data = torch.load(path,weights_only=True)
    if corruption == 'missing':
        path.unlink()
    elif corruption == 'duplicate':
        data['step_id'] = data['forward_id'] = 1
        torch.save(data,path)
    elif corruption == 'heads':
        data['global_head_ids'] = [1,0]
        torch.save(data,path)
    elif corruption == 'rank':
        data['rank_agreement_sha256'] = ['wrong','wrong']
        torch.save(data,path)
    else:
        bits = np.unpackbits(data['mask_bits'].numpy(),bitorder='little')
        bits[0] = 0
        data['mask_bits'] = torch.from_numpy(np.packbits(bits,bitorder='little'))
        torch.save(data,path)
    out = tmp_path/'export'
    with pytest.raises(ValueError):
        prepare.convert(source,out,prompts=1,steps=2,layers=2,heads=2)
    assert not out.exists()
    assert not out.with_name('export.incomplete').exists()


def test_token_mapping_must_match_tile_sizes(source):
    path = source/'prompt_00/trace/layouts/layout.pt'
    data = torch.load(path,weights_only=True)
    data['packed_token_to_tile'][0] = 1
    torch.save(data,path)
    with pytest.raises(ValueError,match='Token/tile counts'):
        prepare.read_prompt(source,0,steps=2,layers=2,heads=2)


def test_cross_prompt_schedule_mismatch_is_fatal(source, tmp_path):
    import shutil
    shutil.copytree(source/'prompt_00',source/'prompt_01')
    path = source/'prompt_01/trace/samples/sample_0/sample.json'
    data = json.loads(path.read_text())
    data['video_sigmas'][1] = .6
    path.write_text(json.dumps(data))
    output = tmp_path/'export'
    with pytest.raises(ValueError,match='actual schedule differs'):
        prepare.convert(source,output,prompts=2,steps=2,layers=2,heads=2)
    assert not output.exists()
