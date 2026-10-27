"""Independent token enumeration protects work, dedup, refresh and send accounting."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/analysis/h3_loadbalance'))
from policies import Layout, evaluate, residency, migration, owners, video_layout, refine


@pytest.mark.parametrize('replica', [False, True])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('mode', ['union', 'separate'])
def test_token_oracle(replica, groups, mode):
    layout = Layout(np.array([1, 2, 2, 3]), np.array([0,1,1,2,2,3,3,3]), np.array([[0,0,0],[1,0,0]]), 2)
    owner = np.array([0,1,2,3,0,1,2,3])
    mask = np.array([[True, False], [True, True]])
    result = evaluate(mask, layout, owner, 4, replica, groups)
    res = residency(layout, owner, 4, replica, groups)
    work, send, recv = np.zeros(4,int), np.zeros(4,int), np.zeros(4,int)
    refresh = 0
    # Enumerate individual Q/K token pairs independently of vectorized tile logic.
    for g in range(4):
        prefix_need, video_need = set(), set()
        for qi, qt in enumerate(layout.tokens):
            if owner[qi] != g:
                continue
            for ki, kt in enumerate(layout.tokens):
                allowed = qt < 2 or kt < 2 or mask[qt-2, kt-2]
                if allowed:
                    work[g] += 1
                    (prefix_need if qt < 2 else video_need).add(ki)
        phases = [prefix_need | video_need] if mode == 'union' else [prefix_need, video_need]
        for needed in phases:
            for k in needed:
                if not res[k,g]:
                    send[owner[k]] += 512
                    recv[g] += 512
        for k in range(layout.n):
            if res[k,g] and owner[k] != g:
                send[owner[k]] += 512
                recv[g] += 512
                refresh += 512
    np.testing.assert_array_equal(result['work'][0],work)
    np.testing.assert_array_equal(result[mode+'_send'][0],send)
    np.testing.assert_array_equal(result[mode+'_recv'][0],recv)
    assert result['refresh_recv'].sum() == refresh


@pytest.mark.parametrize('ranks', [2,4,8])
@pytest.mark.parametrize('geometry', ['temporal','spatial','3d'])
def test_conservation_and_dense_prefix_identity(ranks,geometry):
    coords=np.array(list(np.ndindex(10,2,2)))
    sizes=np.array([8,8]+[4]*40)
    layout=Layout(sizes,np.repeat(np.arange(42),sizes),coords,2)
    v=video_layout(layout,ranks,geometry)
    np.testing.assert_array_equal(np.bincount(v),np.full(ranks,40//ranks))
    own=owners(layout,ranks,v)
    mask=np.eye(40,dtype=bool)
    for groups in [1,ranks]:
        result=evaluate(mask,layout,own,ranks,True,groups)
        assert result['union_recv'].sum() == (ranks-1)*layout.n*512
        send,recv=migration(layout,own,own,ranks)
        assert send.sum()==recv.sum()==0


def test_migration_and_no_refinement_with_prohibitive_penalty():
    sizes=np.array([8]+[4]*4)
    layout=Layout(sizes,np.repeat(np.arange(5),sizes),np.array(list(np.ndindex(4,1,1))),1)
    a=owners(layout,2,np.array([0,0,1,1]))
    b=owners(layout,2,np.array([1,0,0,1]))
    s,r=migration(layout,a,b,2,bytes_per_token=10)
    assert s.sum()==r.sum()==80
    prior=np.eye(4)
    v=np.array([0,0,1,1])
    np.testing.assert_array_equal(refine(prior,layout,2,v,v,migration_weight=1e6),v)


@pytest.mark.parametrize('groups', [1,2,4,8])
def test_audio_group_residency_has_one_collective_copy_per_group(groups):
    sizes=np.array([8,16,8])
    layout=Layout(sizes,np.repeat(np.arange(3),sizes),np.array([[0,0,0]]),2)
    own=np.arange(layout.n)%8
    res=residency(layout,own,8,True,groups)
    assert np.all(res[layout.tokens==0].sum(1)==8)
    audio=res[layout.tokens==1]
    assert np.all(audio.sum(1)==groups)
    for g in range(groups):
        assert np.all(audio[:,g*(8//groups):(g+1)*(8//groups)].sum(1)==1)
    assert np.all(res[np.arange(layout.n),own])
    assert np.all(res[layout.tokens==2].sum(1)==1)
