"""Reject corrupted completion, identity, cache and I/O evidence independently."""
import copy,json
from pathlib import Path
import pytest
from audit import audit
from support import raw,counter_delta,EVENT_BYTES,histogram


def fixture(root):
    stamps=list(range(1,10001));files=['/ffb/a','/ffb/b','/ffb/c','/ffb/d','/ffb/e']
    modules={'psana':dict(path='/pr/psana/psana/__init__.py',sha256='pythonhash'),
             'psana.dgram':dict(path='/install/dgram.so',sha256='nativehash')}
    p=dict(mount={'filesystems':[{'source':'drpsrcf/ffb'}]},gpu_uuid='GPU-test',cpu_affinity=list(range(24)),source_sha256={'harness':'hash'},native_receipt=dict(status='pass',pr_sha='67c327a148a08b2aaf972e67e1e02e367c2b298f',source='/pr',python={'__init__.py':'pythonhash'},native={'/install/dgram.so':'nativehash'}))
    info=dict(timestamps=stamps,timestamp_sha256=raw.ts_hash(stamps),files=files)
    device=dict(mask='GPU-test',name='NVIDIA L40S',compat_mode='ON',effective=dict(num_threads=8,task_size=1<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False))
    data={'provenance.json':p,'inputs.json':info,'gpu-tests.json':dict(exitcode=0,passed=31,failed=0,skipped=0),'cleanup.json':dict(removed=True)}
    matrix=[('preflight-stream1',1,1000),('preflight-stream2',2,1000)]+[('%02d-stream%d'%(i,s),s,10000) for i,s in enumerate((1,2,2,1))]
    for name,streams,n in matrix:
        before={'eth0':dict(monotonic=1.,rx_bytes_phy=100,tx_bytes_phy=10,device='/pci/1')}
        after={'eth0':dict(monotonic=12.,rx_bytes_phy=100+n*EVENT_BYTES,tx_bytes_phy=20,device='/pci/1')}
        idle_after={'eth0':dict(monotonic=2.,rx_bytes_phy=110,tx_bytes_phy=11,device='/pci/1')}
        m=dict(seconds=10.,conditioning=[dict(path=f,resident_pages=0) for f in files],initial_cache=[dict(path=f,resident_pages=0) for f in files],nic_before=before,nic_after=after,nic_delta=counter_delta(before,after,n*EVENT_BYTES),idle=dict(before=before,after=idle_after,delta=counter_delta(before,idle_after)))
        rows=[dict(role=role,affinity=list(range(24)),numa_policy='policy: interleave',runtime_files=copy.deepcopy(modules),timestamps=[]) for role in ('smd0','eb','bd')]
        rows[2].update(timestamps=stamps[:n],callbacks=[dict(size=100,stream=i%streams,timestamps=stamps[i*100:(i+1)*100]) for i in range(n//100)],io=dict(requests=5*n,completed_requests=5*n,requested_bytes=n*EVENT_BYTES,completed_bytes=n*EVENT_BYTES,read_size_histogram=histogram(n),pressure_drains=0),cpu_reads=0,drain=[dict(closed=True,pending=0,active=0,held=0,peers=1)],device=device,memory={})
        data[name+'.json']=dict(status='pass',events=n,bytes=n*EVENT_BYTES,rows=rows,measurement=m,host='ada',framework=dict(batch_size=100,streams=streams,budget_gib=6,bulk_target_bytes=524288,smd_events=1000),source_sha256=p['source_sha256'],seconds=10.,gib_s=n*EVENT_BYTES/2**30/10,hz=n/10)
    data['complete.json']=dict(status='pass',cases=[dict(name=name,streams=s,events=n) for name,s,n in matrix])
    def save():
        for name,value in data.items():(root/name).write_text(json.dumps(value))
    save();return data,save


def test_accept_complete_evidence(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['status']=='pass'


@pytest.mark.parametrize('kind',['gpu','matrix','streams','reads','drain','assigned_gpu','cache','nic','harness','complete','cleanup','python_hash','native_hash'])
def test_reject_corrupted_evidence(tmp_path,kind):
    d,save=fixture(tmp_path);r=d['00-stream1.json'];bd=r['rows'][2]
    if kind=='gpu':d['gpu-tests.json']['skipped']=1
    elif kind=='matrix':d['complete.json']['cases'].pop()
    elif kind=='streams':bd['callbacks'][1]['stream']=1
    elif kind=='reads':bd['io']['completed_requests']-=1
    elif kind=='drain':bd['drain'][0]['held']=1
    elif kind=='assigned_gpu':d['provenance.json']['gpu_uuid']='wrong'
    elif kind=='cache':r['measurement']['initial_cache'][0]['resident_pages']=1
    elif kind=='nic':r['measurement']['nic_delta']['eth0']['rx_bytes']+=1
    elif kind=='harness':r['source_sha256']={'different':'hash'}
    elif kind=='complete':d['complete.json']['status']='incomplete'
    elif kind=='cleanup':d['cleanup.json']['removed']=False
    else:
        # Corrupt all rows consistently so equality across ranks/cases cannot
        # conceal a mismatch against the independently frozen runtime receipt.
        module='psana' if kind=='python_hash' else 'psana.dgram'
        for item in d.values():
            for row in item.get('rows',[]):row['runtime_files'][module]['sha256']='wrong'
    save()
    with pytest.raises(ValueError):audit(tmp_path)
