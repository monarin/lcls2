"""Plan, one-shot deploy, and verify eight experiment-only run-3/4 constants."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import requests

from fixture import HERE, ROOT, SNAPSHOT, frozen_report, create_root, payload_hash, wire, check_stat

RO='https://pswww.slac.stanford.edu/ws/calib_ws/'
RW='https://pswww.slac.stanford.edu/ws-kerb/calib_ws/'
DB='cdb_prjdatpsana'
COLLECTION='jungfrau_000010'
PLAN=HERE/'deployment_plan.json'
EXECUTION=HERE/'deployment'
PLACEHOLDER='DESTINATION_DATA_ID'
ALLOWED={RW+DB+'/gridfs/', RW+DB+'/'+COLLECTION+'/'}


def save(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def body_hash(kwargs):
    if kwargs.get('json') is not None:
        return wire.sha(json.dumps(kwargs['json'],sort_keys=True,separators=(',',':')).encode())
    return wire.sha(kwargs['data'])


class Guard:
    def __init__(self):
        self.original=requests.sessions.Session.request
        self.pending=None

    def install(self):
        guard=self
        def request(session,method,url,*args,**kwargs):
            method=method.upper()
            if method!='GET':
                wire.require(method=='POST' and url in ALLOWED and guard.pending==(url,body_hash(kwargs)), 'Disallowed calibration mutation')
                guard.pending=None  # Single use, even if the request fails.
                kwargs['allow_redirects']=False
            kwargs.setdefault('timeout',180)
            return guard.original(session,method,url,*args,**kwargs)
        requests.sessions.Session.request=request


def get(url,**kwargs):
    r=requests.get(url,**kwargs); r.raise_for_status(); return r.json()


def collection_docs(db,col):
    return sorted(get(RO+db+'/'+col),key=lambda d:d['_id'])


def destination():
    return {c:collection_docs(DB,c) for c in get(RO+DB) if not c.startswith(('fs.','system.'))}


def protected():
    result={}
    for db,col in (('cdb_xpp101605526',COLLECTION),('cdb_'+COLLECTION,COLLECTION)):
        docs=collection_docs(db,col)
        result[db+'/'+col]=dict(count=len(docs),sha256=wire.sha(json.dumps(docs,sort_keys=True).encode()))
    return result


def old_selections(snapshot):
    from psana.pscalib.calib import MDBWebUtils as wu
    result={}
    for col,docs in snapshot.items():
        for run in (1,2):
            result[f'{col}/run{run}']={}
            for ct in sorted({d['ctype'] for d in docs}):
                selected=wu.select_doc_in_run_range([d for d in docs if d['ctype']==ct],run)
                wire.require(selected is not None,'Missing existing run-1/2 selection')
                result[f'{col}/run{run}'][ct]=selected['_id']
    return result


def check_existing(expected, current):
    wire.require({k:v for k,v in current.items() if k!=COLLECTION}==expected, 'Existing destination documents changed')
    for r in frozen_report('admin_checks')['run1_2_files_before'].values():
        check_stat(r)


def prepare():
    from krtc import KerberosTicket
    from psana.pscalib.calib import MDBUtils as mu, MDBWebUtils as wu, CalibConstants as cc
    wire.require(cc.URL==RO and cc.URL_KRB==RW,'Unexpected calibration endpoint')
    create_root()
    source=frozen_report('calibration')
    wire.require(source['shortname']==COLLECTION and len(source['records'])==8,'Wrong frozen detector')
    wire.require(wu.pro_detector_name(source['uniqueid'],add_shortname=False)==COLLECTION,'Detector name mapping changed')
    baseline=json.loads((SNAPSHOT/'destination_calibration_before.json').read_text())
    wire.require(destination()==baseline,'Destination changed since Stage 1 or target collection already exists')
    check_existing(baseline,baseline)
    headers=KerberosTicket('HTTP@pswww.slac.stanford.edu').getAuthHeaders()
    privilege=get(RW+DB+'/test_post_privilege',headers=headers)
    wire.require(privilege.get('status') is True,'No destination write privilege')
    lgbk='https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/prjdatpsana/ws/'
    runs=get(lgbk+'runs',headers=headers)
    wire.require(runs.get('success') is True and not any(r['num'] in (3,4) for r in runs['value']), 'Run 3/4 already registered')
    from fixture import DEST
    wire.require(not [p for folder in (DEST,DEST/'smalldata') for n in (3,4) for p in folder.glob(f'*r{n:04d}*.xtc2*')], 'Published run-3/4 file collision')
    now=int(time.time())
    items=[]
    for record in source['records']:
        raw=(SNAPSHOT/record['file']).read_bytes()
        wire.require(wire.sha(raw)==record['sha256'],'Frozen calibration changed')
        data=mu.object_from_data_string(raw,record['document'])
        wire.require(payload_hash(data)==record['sha256'],'Encoding changes calibration bytes')
        doc=mu.docdic(data,PLACEHOLDER,experiment='prjdatpsana',run=3,run_end=4,
            detector=COLLECTION,shortname=COLLECTION,longname=source['uniqueid'],detname='jungfrau1M',dettype='jungfrau',
            ctype=record['document']['ctype'],dtype=record['document']['dtype'],
            time_sec=now,time_stamp=datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z'),
            version='V2026-11-05',comment='Frozen XPP run-311 JF1M calibration for synthetic prjdatpsana runs 3-4.',
            extpars=dict(source_db=record['source_db'],source_document_id=record['document']['_id'],
                         source_data_id=record['document']['id_data'],source_payload_sha256=record['sha256'],
                         source_events=dict(experiment='xpp101605526',run=311),source_run=record['document']['run'],
                         stage1_calibration_sha256=wire.file_sha(HERE.parent/'synthetic_xpp_stage1_20261105/calibration.json')))
        wire.require(not {'_id','tsec_id','tstamp_id'} & set(doc) and doc['id_doc_exp']==doc['id_data_exp']==0,'Foreign document metadata')
        for field in ('data_type','data_dtype','data_shape','data_size'):
            wire.require(doc.get(field)==record['document'].get(field),'Encoding metadata differs: '+field)
        items.append(dict(file=record['file'],bytes=len(raw),sha256=record['sha256'],document=doc))
    save(PLAN,dict(database=DB,collection=COLLECTION,run=3,run_end=4,items=items,
        protected_before=protected(),existing_before=baseline,existing_selections=old_selections(baseline),
        privilege=privilege,script_sha256=wire.file_sha(HERE/'calibration.py')))
    print('PASS deployment plan:',len(items),'documents,',sum(i['bytes'] for i in items),'bytes; SHA256',wire.file_sha(PLAN),flush=True)


def deploy(guard,expected_sha):
    from krtc import KerberosTicket
    wire.require(expected_sha and wire.file_sha(PLAN)==expected_sha,'Deployment plan not the reviewed hash')
    plan=json.loads(PLAN.read_text())
    wire.require(plan['script_sha256']==wire.file_sha(HERE/'calibration.py'),'Deployment code changed since plan')
    wire.require(destination()==plan['existing_before'],'Destination no longer matches preflight')
    wire.require(protected()==plan['protected_before'],'Source/shared metadata changed')
    check_existing(plan['existing_before'],destination())
    # Validate every frozen payload before the first write.
    for item in plan['items']:
        wire.require(wire.file_sha(SNAPSHOT/item['file'])==item['sha256'],'Frozen payload changed')
    headers=KerberosTicket('HTTP@pswww.slac.stanford.edu').getAuthHeaders()
    wire.require(get(RW+DB+'/test_post_privilege',headers=headers).get('status') is True,'Write privilege lost')
    EXECUTION.mkdir(exist_ok=False)  # Any prior attempt requires inspection, never retry blindly.
    save(EXECUTION/'attempt.json',dict(plan_sha256=expected_sha,started_utc=datetime.now(timezone.utc).isoformat()))
    journal=[]
    for n,item in enumerate(plan['items']):
        results=[]
        for kind in ('data','document'):
            if kind=='data':
                url=RW+DB+'/gridfs/'
                kwargs=dict(data=(SNAPSHOT/item['file']).read_bytes())
            else:
                url=RW+DB+'/'+COLLECTION+'/'
                doc=dict(item['document'],id_data=results[0])
                kwargs=dict(json=doc)
            digest=body_hash(kwargs)
            prefix=EXECUTION/f'{n:02d}-{kind}'
            save(prefix.with_suffix('.intent.json'),dict(url=url,method='POST',ctype=item['document']['ctype'],body_sha256=digest,
                                                       utc=datetime.now(timezone.utc).isoformat()))
            guard.pending=(url,digest)
            response=requests.post(url,headers=headers,**kwargs)
            result=response.json()
            save(prefix.with_suffix('.response.json'),dict(status=response.status_code,response=result))
            response.raise_for_status()
            wire.require(200<=response.status_code<300 and isinstance(result.get('_id'),str),'Missing upload ID')
            results.append(result['_id'])
        journal.append(dict(ctype=item['document']['ctype'],id_data=results[0],id_doc=results[1],sha256=item['sha256']))
        save(EXECUTION/f'{n:02d}-completed.json',journal[-1])
        print('Deployed',item['document']['ctype'],results[1],flush=True)
    save(EXECUTION/'journal.json',journal)
    print('Completed eight experiment-only uploads; readback is required',flush=True)


def verify():
    from psana.pscalib.calib import MDBWebUtils as wu
    plan=json.loads(PLAN.read_text());journal=json.loads((EXECUTION/'journal.json').read_text())
    wire.require(wire.file_sha(PLAN)==json.loads((EXECUTION/'attempt.json').read_text())['plan_sha256'],'Plan changed')
    current=destination();check_existing(plan['existing_before'],current)
    wire.require(protected()==plan['protected_before'],'Source/shared metadata changed')
    wire.require(old_selections({k:v for k,v in current.items() if k!=COLLECTION})==plan['existing_selections'],'Existing selections changed')
    docs=current[COLLECTION]
    wire.require(len(docs)==len(journal)==len(plan['items'])==8,'Unexpected destination document count')
    ids={x['ctype']:x for x in journal}
    for item in plan['items']:
        row=ids[item['document']['ctype']]
        doc=next(d for d in docs if d['_id']==row['id_doc'])
        wire.require(doc==dict(item['document'],id_data=row['id_data'],_id=row['id_doc']),'Destination document differs from planned metadata')
    source=frozen_report('calibration');checks=[]
    for run in (3,4):
        constants=wu.calib_constants_all_types(source['uniqueid'],exp='prjdatpsana',run=run)
        wire.require(set(constants)==set(ids),'Wrong destination calibration type set')
        for ct,(data,doc) in constants.items():
            wire.require(doc['_id']==ids[ct]['id_doc'] and doc['experiment']=='prjdatpsana' and doc['run']==3 and doc['run_end']==4,'Wrong selection/validity')
            wire.require(payload_hash(data)==ids[ct]['sha256'],'Payload readback differs')
            checks.append(dict(run=run,ctype=ct,id_doc=doc['_id'],sha256=ids[ct]['sha256']))
        del constants;gc.collect()
    for ct in ids:
        for excluded in (1,2,5):
            wire.require(wu.select_doc_in_run_range([d for d in docs if d['ctype']==ct],excluded) is None,'New experiment record leaks outside 3-4')
    save(HERE/'calibration_readback.json',dict(status='pass',checks=checks,document_count=8,post_requests=16,
        existing_documents_unchanged=13,existing_run1_2_selections_unchanged=True,protected_metadata_unchanged=True,
        new_experiment_records_excluded_for_runs=[1,2,5],payload_bytes=sum(i['bytes'] for i in plan['items']),
        script_sha256=wire.file_sha(HERE/'calibration.py')))
    print('PASS run-3/4 normal calibration lookup, payload readback and preservation',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','deploy','verify'));p.add_argument('--plan-sha');args=p.parse_args()
    guard=Guard();guard.install()
    if args.action=='prepare':prepare()
    elif args.action=='deploy':deploy(guard,args.plan_sha)
    else:verify()
