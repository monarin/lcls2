"""Validate native serial bridge, focused diagnostic and same-node read control."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
from audit import check,need,sha

HERE=Path(__file__).resolve().parents[1]

def gate(m,kind):
    c=m['conditioning']['cache']
    need(len(c['files'])==10 and sum(x['resident_pages'] for x in c['files'])==c['resident_pages'],'Cache accounting')
    need(c['resident_pages']==0 if kind=='cold' else c['resident_pages']/c['pages']>=.999,'Cache gate')
    need(m['nic_before'].keys()==m['nic_after'].keys()==m['nic_delta'].keys(),'NIC inventory')
    for name,a in m['nic_before'].items():
        b=m['nic_after'][name];d=m['nic_delta'][name]
        rx=b['rx_bytes_phy']-a['rx_bytes_phy'];tx=b['tx_bytes_phy']-a['tx_bytes_phy'];dt=b['monotonic']-a['monotonic']
        need(a['device']==b['device'] and rx>=0 and tx>=0 and dt>0,'NIC reset')
        need(d['rx_bytes']==rx and d['tx_bytes']==tx and d['seconds']==dt,'NIC deltas')
        need(math.isclose(d['rx_gib_s'],rx/(1<<30)/dt,rel_tol=1e-12),'NIC rate')


def audit(main,serial,detailed):
    records=[];artifacts={};sources={}
    reference=json.loads((main/'full-b1-s2-cold-r1.json').read_text())
    b=reference['rows'][2];host=b['host'];mask=b['runtime']['device_mask']
    for kind in ('cold','warm'):
        np=serial/('serial-'+kind)/'network.json';rp=np.with_name('report.json')
        n=json.loads(np.read_text());r=json.loads(rp.read_text());gate(n['measurement'],kind)
        need(n['status']==r['status']==r['checks']['status']=='pass','Serial failed')
        need(r['host']==host and r['runtime']['environment']['CUDA_VISIBLE_DEVICES']==mask,'Serial placement')
        need(r['delivered_events']==r['checks']['events']==10000 and sum(c['size'] for c in r['callbacks'])==10000,'Serial count')
        need(r['bank_events']==[5000,5000] and r['checks']['counts']==[102]+[101]*98+[0],'Serial bins')
        need(r['admission_pressure_trims']==dict(warmup=0,steady=0),'Serial drains')
        need(n['io']['completed_bytes']==n['io']['requested_bytes']==33582000000 and n['io']['completed_requests']==n['io']['requests']==50000 and n['io']['pressure_drains']==0,'Serial I/O')
        need(n['native_sha256']==sha(rp),'Native hash')
        need(n['iteration_seconds']==r['iteration_completed_seconds'] and math.isclose(n['jf_gib_s'],33582000000/(1<<30)/n['iteration_seconds'],rel_tol=1e-12),'Serial rate')
        dt=n['measurement']['envelope_end']-n['measurement']['envelope_begin']
        need(0<=dt-n['iteration_seconds']<.1,'Serial bracket mismatch')
        for src,h in {**n['source_sha256'],**r['code_sha256']}.items():
            need(sha(src)==h,'Changed serial source');sources[src]=h
        records.append(dict(mode='serial',cache=kind,jf_gib_s=n['jf_gib_s'],seconds=n['iteration_seconds'],
            steady_events_per_second=r['window']['completed_events_per_second'],nic=n['measurement']['nic_delta']))
        artifacts[str(np)]=sha(np);artifacts[str(rp)]=sha(rp)
        p=serial/('mpi-'+kind+'.json');r=json.loads(p.read_text());check(r)
        need(r['rows'][2]['host']==host and r['rows'][2]['runtime']['device_mask']==mask,'MPI bridge placement')
        need(r['rows'][2]['timestamps']==reference['rows'][2]['timestamps'],'MPI bridge events')
        records.append(dict(mode='mpi',cache=kind,jf_gib_s=r['jf_gib_s'],seconds=r['measurement']['loop_seconds'],nic=r['measurement']['nic_delta']))
        artifacts[str(p)]=sha(p)
    p=detailed/'report.json';r=json.loads(p.read_text());bds=check(r);extra=r['additional_instrumentation']
    need(sha(extra['path'])==extra['sha256'],'Changed diagnostic wrapper')
    need(r['rows'][2]['host']==host and r['rows'][2]['runtime']['device_mask']==mask,'Diagnostic placement')
    need(r['rows'][2]['timestamps']==reference['rows'][2]['timestamps'],'Diagnostic events')
    artifacts[str(p)]=sha(p);sources[extra['path']]=extra['sha256']
    spec=importlib.util.spec_from_file_location('read_audit',HERE.parent/'mfx_kvikio_read_20261108/audit.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    p=main/'standalone.json';standalone=json.loads(p.read_text());reader.check_case(standalone)
    need(standalone['host']==host and standalone['passes'][0]['network']['cache_before']['resident_pages']==0,'Standalone cold node/gate')
    artifacts[str(p)]=sha(p)
    return dict(status='pass',host=host,device_mask=mask,artifacts=artifacts,source_verified=sources,serial_bridge=records,
        detailed=dict(loop_seconds=r['measurement']['loop_seconds'],rank_seconds=bds[0]['iteration_seconds'],profile=bds[0]['profile'],nic=r['measurement']['nic_delta']),
        standalone=dict(config=standalone['config'],passes=standalone['passes']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',type=Path,required=True);p.add_argument('--serial',type=Path,required=True);p.add_argument('--detailed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(audit(a.main,a.serial,a.detailed),f,indent=2);f.write('\n')
