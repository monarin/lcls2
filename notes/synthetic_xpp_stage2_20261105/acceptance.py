"""Normal serial DataSource acceptance; source references, raw pilot, calibrated pilot."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from fixture import HERE, ROOT, SOURCE, RUN, fingerprint, frozen_report, load_sources, plan, payload_hash, wire
from common import guard_http

SAMPLES=(0,1,2,10,1199,2399)


def main(kind,batch_size):
    guard_http()
    wire.require(os.environ.get('PS_FAKESTEP_FLAG')=='0' and os.environ.get('PS_PARALLEL')=='none','Wrong runtime flags')
    from psana import DataSource
    import psana
    source_kind=kind=='source'
    raw_only=kind.endswith('_raw')
    mode=kind.removesuffix('_raw') if not source_kind else 'pilot'
    sources=load_sources()
    mapping=plan(sources[0]['index'],mode)
    source_index=sources[0]['index']
    source_event_rows=np.flatnonzero(source_index['service']==12)
    source_event_number={int(row):n for n,row in enumerate(source_event_rows)}
    out_event_rows=np.flatnonzero(source_index['service'][mapping['source_record']]==12)
    expected_times=mapping['timestamp'][out_event_rows]
    kwargs=dict(exp='xpp101605526' if source_kind else 'prjdatpsana',run=311 if source_kind else RUN,
                dir=str(SOURCE if source_kind else ROOT/mode),live=False,batch_size=batch_size)
    if source_kind:
        kwargs.update(timestamps=source_index['timestamp'][source_event_rows[list(SAMPLES)]],detectors=['jungfrau1M'])
    elif raw_only:
        kwargs['skip_calib_load']='all'
    ds=DataSource(**kwargs)
    constants=frozen_report('calibration')
    deployed={} if source_kind or raw_only else {r['ctype']:r for r in json.loads((HERE/'deployment/journal.json').read_text())}
    reference=None if source_kind else json.loads((HERE/'source_samples.json').read_text())
    samples={};checks=[];counts=[];cursor=0;runs=0;detnames=[]
    for run in ds.runs():
        runs+=1
        wire.require((run.expt,run.runnum)==(kwargs['exp'],kwargs['run']),'Wrong run identity')
        jf=run.Detector('jungfrau1M');detnames=sorted(run.detnames)
        wire.require(list(jf.raw._sorted_segment_inds)==[0,1],'Wrong JF segments')
        if not source_kind:
            wire.require('timing' in run.detnames and not {'alvium_1','epix100','alvium_tt'} & set(run.detnames),'Wrong selected detectors')
        if not raw_only:
            wire.require(set(jf.calibconst)=={r['document']['ctype'] for r in constants['records']},'Wrong calibration types')
            for record in constants['records']:
                ct=record['document']['ctype'];data,doc=jf.calibconst[ct]
                expected_id=record['document']['_id'] if source_kind else deployed[ct]['id_doc']
                wire.require(doc['_id']==expected_id and payload_hash(data)==record['sha256'],'Wrong calibration selection/payload')
                if not source_kind:
                    wire.require(doc['experiment']=='prjdatpsana' and (doc['run'],doc['run_end'])==(3,4),'Wrong calibration validity')
                checks.append(dict(ctype=ct,id_doc=doc['_id'],sha256=record['sha256']))

        def consume(evt,source_n):
            raw=jf.raw.raw(evt)
            wire.require(raw is not None and raw.shape==(2,512,1024) and raw.dtype==np.uint16,'Wrong/missing 1M raw frame')
            if source_n in SAMPLES:
                value=dict(raw=fingerprint(raw))
                if not raw_only:
                    calibrated=jf.raw.calib(evt,cversion=3,cmpars=None)
                    wire.require(calibrated is not None and calibrated.shape==raw.shape and np.isfinite(calibrated).all(),'Invalid calibrated output')
                    value['calibrated']=fingerprint(calibrated)
                if reference is not None:
                    for name,fp in value.items():
                        wire.require(fp==reference['samples'][str(source_n)][name],'Source/destination sample differs')
                if str(source_n) in samples:
                    wire.require(value==samples[str(source_n)],'Repeated payload/calibration changed')
                samples[str(source_n)]=value

        if source_kind:
            for n,evt in enumerate(run.events()):
                wire.require(n<len(SAMPLES) and int(evt.timestamp)==int(kwargs['timestamps'][n]),'Wrong source sample order')
                consume(evt,SAMPLES[n]);cursor+=1
            wire.require(cursor==len(SAMPLES),'Missing source samples')
        else:
            step_time=int(mapping['timestamp'][np.flatnonzero(source_index['service'][mapping['source_record']]==6)[0]])
            for step in run.steps():
                wire.require(not counts and int(step.evt.timestamp)==step_time,'Unexpected step/time')
                count=0
                for evt in step.events():
                    wire.require(cursor<len(expected_times) and int(evt.timestamp)==int(expected_times[cursor]),'Lost/duplicated/reordered event')
                    source_row=int(mapping['source_record'][out_event_rows[cursor]])
                    consume(evt,source_event_number[source_row]);count+=1;cursor+=1
                counts.append(count)
            wire.require(counts==[mapping['events']] and cursor==mapping['events'],'Wrong complete-run count')
    wire.require(runs==1,'Wrong run count')
    expected_samples=set(SAMPLES) if source_kind or mode=='pilot' else {0,1,2}
    wire.require(set(map(int,samples))==expected_samples,'Missing acceptance samples')
    report=dict(status='pass',kind=kind,events=cursor,step_counts=counts,detectors=detnames,samples=samples,
                calibration=checks,batch_size=batch_size,PS_SMD_N_EVENTS=os.environ.get('PS_SMD_N_EVENTS'),
                PS_FAKESTEP_FLAG=os.environ['PS_FAKESTEP_FLAG'],psana=psana.__file__,
                script_sha256=wire.file_sha(HERE/'acceptance.py'))
    wire.save_json(HERE/('source_samples.json' if source_kind else f'{kind}_acceptance.json'),report)
    print('PASS DataSource',kind,'steps',counts,'events',cursor,'sample indices',sorted(samples),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('source','pilot_raw','seam_raw','pilot','seam'));p.add_argument('--batch-size',type=int,default=37)
    a=p.parse_args();main(a.kind,a.batch_size)
