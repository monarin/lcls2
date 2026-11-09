"""Independent event, byte, cache, NIC and scaling summary (no CUDA needed)."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

GIB=1<<30

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(r):
    need(r['status']=='pass','Failed case');s=r['settings'];m=r['measurement'];bds=[x for x in r['rows'] if x['role']=='bd']
    need(len(bds)==s['bds'] and len(r['rows'])==s['bds']+2,'Rank topology')
    need(len({x['host'] for x in r['rows']})==1 and len({x['job'] for x in r['rows']})==1,'Multiple nodes/jobs')
    need(s['batch']==7 and s['bds']*s['streams']<=4,'Unexpected batch or state budget')
    need([x['role'] for x in r['rows'][:2]]==['smd0','eb'],'Helper roles')
    expected=r['events']
    delivered=[t for b in bds for t in b['timestamps']]
    need(len(delivered)==expected and len(set(delivered))==expected,'Missing/duplicate global events')
    need(r['jf_bytes']==expected*3358200,'Global bytes')
    need(math.isclose(r['jf_gib_s'],r['jf_bytes']/GIB/m['loop_seconds'],rel_tol=1e-12),'Rate arithmetic')
    need(math.isclose(r['events_per_second'],expected/m['loop_seconds'],rel_tol=1e-12),'Event rate arithmetic')
    cpus=[c for row in r['rows'] for c in row['cpu_affinity']]
    need(len(set(cpus))==len(cpus) and all(len(b['cpu_affinity'])==4 for b in bds),'Overlapping CPU sets')
    need(len({b['runtime']['device_pci'] for b in bds})==1 and len({b['runtime']['device_mask'] for b in bds})==1,'GPU placement')
    cold=m['conditioning']['cache'];after=m['cache_after']
    for cache in (cold,after):
        need(len(cache['files'])==10 and len({f['path'] for f in cache['files']})==10,'Cache file coverage')
        need(sum(f['pages'] for f in cache['files'])==cache['pages'] and sum(f['resident_pages'] for f in cache['files'])==cache['resident_pages'],'Cache accounting')
        need(all(0<=f['resident_pages']<=f['pages'] for f in cache['files']),'Invalid residency')
    fraction=cold['resident_pages']/cold['pages']
    need(fraction==0 if s['cache']=='cold' else fraction>=.999,'Cache gate')
    keys=m['nic_before'].keys()
    need(keys==m['nic_after'].keys()==m['nic_delta'].keys()==m['idle']['before'].keys()==m['idle']['after'].keys(),'NIC inventory changed')
    for name in keys:
        a=m['nic_before'][name];b=m['nic_after'][name];d=m['nic_delta'][name]
        seconds=b['monotonic']-a['monotonic'];rx=b['rx_bytes_phy']-a['rx_bytes_phy'];tx=b['tx_bytes_phy']-a['tx_bytes_phy']
        need(b['device']==a['device'] and rx>=0 and tx>=0 and seconds>=m['loop_seconds'],'NIC endpoints')
        need(d['rx_bytes']==rx and d['tx_bytes']==tx and d['seconds']==seconds,'NIC delta arithmetic')
        need(math.isclose(d['rx_gib_s'],rx/GIB/seconds,rel_tol=1e-12),'NIC rate')
    total_counts=[0]*100
    global_order={t:i for i,t in enumerate(sorted(delivered))}
    for b in bds:
        n=len(b['timestamps']);io=b['io'];runtime=b['runtime'];checks=b['checks']
        need(b['timestamps']==sorted(b['timestamps']),'Rank order')
        need(runtime['effective']==dict(num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False) and runtime['compat_mode']=='ON','Effective KvikIO settings')
        need(io['completed_bytes']==io['requested_bytes']==3358200*n and io['completed_requests']==io['requests']==5*n,'Reader accounting')
        need(io['pressure_drains']==0,'Admission drains')
        need([t for c in b['callbacks'] for t in c['timestamps']]==b['timestamps'] and all(c['size']==len(c['timestamps']) for c in b['callbacks']),'Task/delivery identity')
        need(checks['events']==n and checks['status']=='pass','Task acceptance')
        if s['task']=='full':
            need(len(checks['banks'])==s['streams'] and len({c['stream'] for c in b['callbacks']})==s['streams'],'Stream use')
            local=[0]*100
            for t in b['timestamps']:local[global_order[t]%99]+=1
            need(checks['counts']==local,'Wrong local bins')
            need([sum(bank['counts'][i] for bank in checks['banks']) for i in range(100)]==local,'Bank sums')
            total_counts=[a+b for a,b in zip(total_counts,local)]
        if s['diagnostic']:
            prof=b['profile'];exclusive=sum(v['wall_exclusive'] for v in prof['cpu'].values())
            need(0<=exclusive<=b['iteration_seconds']+.1,'Exclusive CPU spans exceed loop')
            need(all(v['wall_exclusive']>=-1e-6 and v['thread_exclusive']>=-1e-6 for v in prof['cpu'].values()),'Negative exclusive spans')
    if s['task']=='full':need(sum(total_counts)==expected and total_counts[-1]==0,'Global bins')
    return bds


def audit(root):
    complete=json.loads((root/'complete.json').read_text());need(complete['status']=='pass','Incomplete campaign')
    expected={'pilot-b4-s1','pilot-b2-s2','diag-full-b1-s2','diag-full-b2-s2','diag-parse-b1-s2'}
    for repeat in (1,2):
        configs=[(1,2,'full'),(2,2,'full'),(4,1,'full')]
        if repeat==1:configs += [(1,1,'full'),(2,1,'full'),(1,2,'parse')]
        for b,s,t in configs:
            expected.update('%s-b%d-s%d-%s-r%d'%(t,b,s,k,repeat) for k in ('cold','warm'))
    names=[x['name'] for x in complete['cases']]
    need(set(names)==expected and len(names)==len(expected),'Missing/duplicate campaign cases')
    cases=[];artifacts={};sources={};baseline_identity=None
    for item in complete['cases']:
        path=root/(item['name']+'.json');r=json.loads(path.read_text());bds=check(r)
        need(item['jf_gib_s']==r['jf_gib_s'] and item['events_per_second']==r['events_per_second'],'Progress/native disagreement')
        # Check all event identities against the accepted manifest, not just count.
        manifest_path=Path('/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_10k/generation.json')
        need(sha(manifest_path)==r['identity']['generation_sha256'],'Manifest identity')
        manifest=json.loads(manifest_path.read_text())
        stamps=[x['timestamp'] for x in manifest['schedule'] if x['service']==12][:r['events']]
        need(sorted(t for b in bds for t in b['timestamps'])==stamps,'Event identities')
        identity=(r['rows'][0]['host'],r['rows'][0]['job'],r['identity'],r['calibration'],bds[0]['runtime'],bds[0]['runtime_files'],r['source_sha256'])
        if baseline_identity is None:baseline_identity=identity
        need(identity==baseline_identity,'Runtime/source/fixture changed')
        for source,h in r['source_sha256'].items():
            need(sha(source)==h,'Source changed '+source);sources[source]=h
        for row in bds:
            for record in row['runtime_files'].values():
                need(sha(record['path'])==record['sha256'],'Runtime source changed')
        artifacts[str(path)]=sha(path)
        m=r['measurement'];s=r['settings']
        cases.append(dict(name=item['name'],bds=s['bds'],streams=s['streams'],task=s['task'],cache=s['cache'],diagnostic=s['diagnostic'],events=r['events'],
            loop_seconds=m['loop_seconds'],events_per_second=r['events_per_second'],jf_gib_s=r['jf_gib_s'],
            nic=m['nic_delta'],cache_before_fraction=m['conditioning']['cache']['resident_pages']/m['conditioning']['cache']['pages'],
            bank_state_gib=bds[0]['memory_before']['aggregate_state_bytes']/GIB,
            event_distribution=[len(b['timestamps']) for b in bds],
            per_bd=[dict(rank=b['rank'],iteration_seconds=b['iteration_seconds'],io=b['io'],profile=b['profile'],memory_after=b['memory_after']) for b in bds]))
    groups=defaultdict(list)
    for c in cases:
        if c['events']==10000 and not c['diagnostic']:groups[(c['task'],c['bds'],c['streams'],c['cache'])].append(c['jf_gib_s'])
    medians=[dict(task=t,bds=b,streams=s,cache=k,repeats=len(v),rates=v,median=statistics.median(v)) for (t,b,s,k),v in groups.items()]
    standalone_path=root/'standalone.json';standalone=json.loads(standalone_path.read_text());need(standalone['status']=='pass','Standalone failed')
    artifacts[str(standalone_path)]=sha(standalone_path)
    return dict(status='pass',host=json.loads((root/(complete['cases'][0]['name']+'.json')).read_text())['rows'][0]['host'],
        artifacts=artifacts,source_verified=sources,cases=cases,medians=medians,
        standalone=dict(runtime=standalone['runtime'],config=standalone['config'],passes=standalone['passes']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(audit(a.root),f,indent=2);f.write('\n')
