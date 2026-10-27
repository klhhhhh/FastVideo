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


@pytest.mark.parametrize('spread', [1,2,4,8])
@pytest.mark.parametrize('groups', [1,2,4,8])
def test_joint_spread_replica_token_oracle(spread, groups):
    sizes=np.array([8,16,3,2,4,1,2,3,1,2])
    layout=Layout(sizes,np.repeat(np.arange(len(sizes)),sizes),np.array(list(np.ndindex(2,2,2))),2)
    own=owners(layout,8,np.arange(8),prefix_spread=spread)
    assert set(own[layout.tokens<2])==set(range(spread))
    mask=np.eye(8,dtype=bool)
    r=evaluate(mask,layout,own,8,True,groups)
    for mode in ['union','separate']:
        assert sum(r[mode+'_'+m+'_demand'][0] for m in ['text','audio','video']) == r[mode+'_demand'][0]
        assert r[mode+'_flows'].sum() == r[mode+'_recv'].sum() == r[mode+'_send'].sum()
    # Independent per-token audio demand for video and prefix phases.
    res=residency(layout,own,8,True,groups)
    total=0
    for g in range(8):
        for prefix in [True,False]:
            queries=[i for i,t in enumerate(layout.tokens) if own[i]==g and (t<2)==prefix]
            if queries:
                total+=sum(512 for k,t in enumerate(layout.tokens) if t==1 and not res[k,g])
    assert total==r['separate_audio_demand'][0]


def test_prefetch_oracle_and_graph_capacity():
    from plan_experiments import prefetch_metrics, graph_layout, pareto
    a=np.zeros((1,40,40),bool); p=a.copy()
    a[0,0,[20,21]]=True;p[0,0,[20,22]]=True
    video=np.arange(40)//20
    result=prefetch_metrics(a,p,np.ones(40,int),video,2)
    assert result['useful_prefetch_bytes']==512
    assert result['wasted_prefetch_bytes']==512
    assert result['residual_remote_bytes']==512
    assert result['recall']==result['precision']==.5
    for method in ['graph','greedy','swap']:
        v=graph_layout(a[0].astype(float),np.ones(40),2,video,method)
        np.testing.assert_array_equal(np.bincount(v),[20,20])
    rows=[dict(x=1,y=3),dict(x=2,y=2),dict(x=3,y=4)]
    assert pareto(rows,['x','y'])==rows[:2]
