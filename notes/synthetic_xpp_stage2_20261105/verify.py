"""Independently verify every output byte except the specifically allowed edits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

import numpy as np

from fixture import HERE, ROOT, RUN, frozen_report, load_sources, plan, wire, check_stat, state_check


def compare_record(original, actual, schemas, timestamp, bd_offset=None, bd_size=None):
    a,b = wire.header(original),wire.header(actual)
    wire.require(len(original)==len(actual) and a[1:]==b[1:], 'Header/extent/control/damage changed')
    wire.require(b[0]==int(timestamp), 'Wrong output timestamp')
    wire.to_ns(b[0])
    svc = (a[1]>>24)&15
    allowed = [(0,8)]
    if svc==4:
        key,schema = wire.schema_for(schemas,'runinfo')
        fields = wire.spans(actual,key,schema)
        p,n,_,_ = fields['expt']
        wire.require(actual[p:p+n]==b'prjdatpsana'.ljust(n,b'\0'), 'Wrong destination experiment')
        allowed.append((p,p+n))
        p,n,dtype,rank = fields['runnum']
        wire.require((n,dtype,rank)==(4,2,0) and struct.unpack_from('<I',actual,p)[0]==RUN, 'Wrong destination run')
        allowed.append((p,p+n))
    if svc==12 and bd_offset is not None:
        (p,off),(q,size) = wire.smd_offsets(actual,schemas)
        wire.require((off,size)==(bd_offset,bd_size), 'Wrong SMD reference')
        allowed.extend(((p,p+8),(q,q+8)))
    cursor=0
    for start,end in sorted(allowed):
        wire.require(original[cursor:start]==actual[cursor:start], 'Unexpected payload change')
        cursor=end
    wire.require(original[cursor:]==actual[cursor:], 'Unexpected trailing payload change')


def verify(mode):
    sources=load_sources()
    mapping=plan(sources[0]['index'],mode)
    folder=ROOT/mode
    summary=json.loads((HERE/f'{mode}_generation.json').read_text())
    wire.require(wire.file_sha(folder/'generation.json')==summary['generation_sha256'], 'Generation manifest changed')
    generation=json.loads((folder/'generation.json').read_text())
    for name,digest in generation['code_sha256'].items():
        wire.require(wire.file_sha(HERE/name)==digest, 'Generator code changed')
    for key in ('source_record','timestamp','cycle'):
        wire.require(generation['mapping'][key]==mapping[key].tolist(), 'Generated provenance mismatch')

    def stream(source):
        item=next(s for s in generation['streams'] if s['stream']==source['stream'])
        files=item['files']
        for k,f in files.items():
            wire.require((folder/f['relative_path']).stat().st_size==f['bytes'] and wire.file_sha(folder/f['relative_path'])==f['sha256'], 'Output file changed')
        with Path(source['entry']['bigdata']['path']).open('rb') as original, \
                (folder/files['bd']['relative_path']).open('rb') as bd, \
                (folder/files['smd']['relative_path']).open('rb') as smd:
            bd_cursor=smd_cursor=0
            services=[]
            for source_record,ts in zip(mapping['source_record'],mapping['timestamp']):
                row=source['index'][source_record]
                source_bd=wire.read_record(original,int(row['bd_offset']))
                p,n=int(row['smd_offset']),int(row['smd_size'])
                source_smd=source['smd'][p:p+n]
                actual_bd=wire.read_record(bd,bd_cursor)
                actual_smd=wire.read_record(smd,smd_cursor)
                compare_record(source_bd,actual_bd,source['bd_schemas'],ts)
                compare_record(source_smd,actual_smd,source['smd_schemas'],ts,bd_cursor,len(actual_bd))
                wire.require(wire.header(actual_bd)[:2]==wire.header(actual_smd)[:2], 'Output stream headers disagree')
                services.append(wire.service(actual_bd))
                bd_cursor+=len(actual_bd); smd_cursor+=len(actual_smd)
            wire.require((bd_cursor,smd_cursor)==(files['bd']['bytes'],files['smd']['bytes']), 'Unaccounted output bytes')
            wire.require(not bd.read(1) and not smd.read(1), 'Unexpected output tail')
        check_stat(source['entry']['bigdata'])
        state=np.empty(len(services),dtype=[('timestamp','<u8'),('service','<u8')])
        state['timestamp'],state['service']=mapping['timestamp'],services
        wire.require(np.all(state['timestamp'][1:]>state['timestamp'][:-1]), 'Non-strict timestamps')
        state_check(state)
        counts=dict(Counter(services))
        wire.require(counts[12]==mapping['events'] and all(counts[n]==1 for n in (2,4,5,6,7,8,9)), 'Wrong event/transition count')
        print('Verified every retained datagram:',mode,'stream',source['stream'],flush=True)
        return dict(stream=source['stream'],counts=counts,records=len(services))

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked=list(pool.map(stream,sources))
    if mode=='pilot':
        expected=frozen_report('layout')['plans'][0]
        wire.require(generation['total_bytes']==expected['total_bytes'], 'Pilot size estimate changed')
        observed={f['relative_path']:f['bytes'] for s in generation['streams'] for f in s['files'].values()}
        wire.require(observed=={f['relative_path']:f['bytes'] for f in expected['files']}, 'Per-file sizes changed')
    else:
        event_rows=sources[0]['index']['service'][mapping['source_record']]==12
        wire.require(mapping['cycle'][event_rows].tolist()==[0,0,0,1,1,1,2,2], 'Seams not exercised')
        event_times=mapping['timestamp'][event_rows]
        wire.require(int(event_times[3])>>32 > int(event_times[2])>>32, 'Seam does not cross second boundary')
    wire.save_json(HERE/f'{mode}_verification.json',dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),
        mode=mode,streams=checked,total_bytes=generation['total_bytes'],every_retained_byte_compared=True,
        timestamp_service_alignment='all four streams match the same independently reconstructed schedule',
        script_sha256=wire.file_sha(HERE/'verify.py')))
    print('PASS',mode,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('pilot','seam'));verify(p.parse_args().mode)
