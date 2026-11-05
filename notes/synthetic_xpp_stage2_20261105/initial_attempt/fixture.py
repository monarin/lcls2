"""Bounded Stage 2 fixture helpers using the checksum-pinned Stage 1 inventory."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parent / 'synthetic_xpp_stage1_20261105'
CHECKSUMS = STAGE1 / 'SHA256SUMS'
if hashlib.sha256(CHECKSUMS.read_bytes()).hexdigest() != 'b7bfe135dc739661e6bc3b5023d66bd04a660377110f65a39a518b30091a166a':
    raise ValueError('Stage 1 checksum manifest changed')
HASHES = dict((name, digest) for digest, name in (line.split('  ', 1) for line in CHECKSUMS.read_text().splitlines()))
for name in ('common.py', 'inventory.py', 'plan_layout.py'):
    if hashlib.sha256((STAGE1 / name).read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError('Stage 1 dependency changed: ' + name)
sys.path.insert(0, str(STAGE1))
from common import wire, check_stat, record_stat, SOURCE, DEST, SELECTED
from inventory import state_check
from plan_layout import schedule as original_schedule

SNAPSHOT = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_xpp_stage1_private_20261105')
ROOT = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_xpp_stage2_private_20261105')
RUN = 3


def frozen_report(name):
    path = STAGE1 / (name + '.json')
    wire.require(wire.file_sha(path) == HASHES[path.name], 'Stage 1 report changed: ' + name)
    return json.loads(path.read_text())


def private_dir(path):
    path.mkdir(mode=0o700, exist_ok=False)
    wire.require(path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o077 == 0, 'Directory is not owner-only')


def create_root():
    os.umask(0o077)
    if not ROOT.exists():
        private_dir(ROOT)
    wire.require(ROOT.stat().st_uid == os.getuid() and ROOT.stat().st_mode & 0o077 == 0, 'Staging root is not private')
    fs = os.statvfs(ROOT)
    wire.require(fs.f_bavail * fs.f_frsize > 6_000_000_000, 'Insufficient pilot staging space')


def private_output(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')


def load_sources():
    inv = frozen_report('inventory')
    sources = []
    for stream in SELECTED:
        entry = inv['streams'][stream]
        check_stat(entry['bigdata'])
        check_stat({k:v for k,v in entry['smd'].items() if k != 'sha256'})
        index_path = SNAPSHOT / 'indexes' / f's{stream:03d}.npy'
        wire.require(wire.file_sha(index_path) == entry['index_sha256'], 'Index changed')
        index = np.load(index_path, allow_pickle=False)
        smd = (SNAPSHOT / 'smd' / Path(entry['smd']['path']).name).read_bytes()
        wire.require(wire.sha(smd) == entry['smd']['sha256'], 'SMD changed')
        with Path(entry['bigdata']['path']).open('rb') as f:
            config = wire.read_record(f, 0)
            wire.require(wire.sha(config) == entry['bd_config_sha256'], 'Configure changed')
            for transition in entry['transitions']:
                svc = transition['service']
                row = index[np.flatnonzero(index['service'] == svc)[0]]
                actual = wire.read_record(f, int(row['bd_offset']))
                wire.require(wire.sha(actual) == transition['sha256'], 'Transition source changed')
                p,n = int(row['smd_offset']), int(row['smd_size'])
                if svc in (4,6,7,8,9):
                    wire.require(actual == smd[p:p+n], 'SMD/bigdata transition mismatch')
                if svc == 5:
                    wire.require(n == 24 and wire.header(smd[p:p+n])[:5] == wire.header(actual)[:5], 'SMD EndRun mismatch')
        sources.append(dict(stream=stream, entry=entry, index=index, smd=smd,
                            bd_schemas=wire.schemas(config), smd_schemas=wire.schemas(smd[:int(index[0]['smd_size'])])))
    for source in sources[1:]:
        wire.require(np.array_equal(source['index'][['timestamp','service']], sources[0]['index'][['timestamp','service']]), 'Source streams differ')
    return sources


def repeat_schedule(index, event_count, source_events):
    """Generalized Stage 1 schedule; compare production maps to frozen outputs."""
    svc = index['service']
    opening = [int(np.flatnonzero(svc == n)[0]) for n in (2,4,6,8)]
    closing = [int(np.flatnonzero(svc == n)[0]) for n in (9,7,5)]
    body = np.arange(opening[-1] + 1, closing[0])
    wire.require(np.all(np.isin(svc[body], [10,12])), 'Unexpected body service')
    wire.require(np.count_nonzero(svc[body] == 12) == source_events and event_count > 0, 'Wrong event count')
    period = wire.to_ns(int(index['timestamp'][closing[0]])) - wire.to_ns(int(index['timestamp'][opening[-1]]))
    rec, ts, cycle = list(opening), list(map(int, index['timestamp'][opening])), [-1]*4
    remain, ncycle = event_count, 0
    while remain:
        take = min(remain, source_events)
        last = int(np.flatnonzero(svc[body] == 12)[take-1])
        selected = body if take == source_events else body[:last+1]
        rec.extend(map(int, selected))
        ts.extend(wire.from_ns(wire.to_ns(int(index['timestamp'][i])) + ncycle*period) for i in selected)
        cycle.extend([ncycle]*len(selected))
        remain -= take
        ncycle += 1
    last = wire.to_ns(ts[-1])
    rec.extend(closing)
    ts.extend(wire.from_ns(last+j*1000) for j in (1,2,3))
    cycle.extend([-1]*3)
    wire.require(all(b>a for a,b in zip(ts,ts[1:])), 'Unordered output schedule')
    state = np.empty(len(rec), dtype=[('timestamp','<u8'),('service','<u8')])
    state['timestamp'], state['service'] = ts, svc[rec]
    state_check(state)
    return np.array(rec,dtype='<u4'), np.array(ts,dtype='<u8'), np.array(cycle,dtype='<i4'), period


def plan(index, mode):
    if mode == 'pilot':
        rec,ts,cyc,period = repeat_schedule(index,2400,151152)
        frozen = frozen_report('layout')['plans'][0]
        mapping = SNAPSHOT / 'run3_mapping.npz'
        wire.require(wire.file_sha(mapping) == frozen['mapping_sha256'], 'Pilot mapping changed')
        with np.load(mapping, allow_pickle=False) as old:
            for key,value in (('source_record',rec),('timestamp',ts),('cycle',cyc)):
                wire.require(np.array_equal(old[key],value), 'Pilot mapping differs from Stage 1')
        return dict(mode=mode, source_record=rec, timestamp=ts, cycle=cyc, events=2400, period_ns=period)
    wire.require(mode == 'seam', 'Only pilot and tiny seam fixture authorized')
    events = np.flatnonzero(index['service'] == 12)
    closing = [int(np.flatnonzero(index['service'] == n)[0]) for n in (9,7,5)]
    # Preserve all body records before the fourth event, including trailing SlowUpdates.
    selected = np.array(list(range(int(events[3]))) + closing, dtype='<u4')
    mini = index[selected].copy()
    last_ns = wire.to_ns(int(mini[-4]['timestamp']))
    mini['timestamp'][-3:] = [wire.from_ns(last_ns+j*1000) for j in (1,2,3)]
    rec,ts,cyc,period = repeat_schedule(mini,8,3)
    # Put the end of the first short body just below a second boundary.
    shift = 999_999_000 - last_ns % 1_000_000_000
    ts = np.array([wire.from_ns(wire.to_ns(int(t))+shift) for t in ts],dtype='<u8')
    return dict(mode=mode, source_record=selected[rec], timestamp=ts, cycle=cyc,
                events=8, period_ns=period, synthetic_short_cycle_events=3, global_shift_ns=shift)


def rewrite(raw, timestamp, schemas, offset=None, size=None):
    out = bytearray(raw)
    struct.pack_into('<Q',out,0,int(timestamp))
    if wire.service(raw) == 4:
        key,schema = wire.schema_for(schemas,'runinfo')
        fields = wire.spans(raw,key,schema)
        p,n,dtype,rank = fields['expt']
        wire.require((dtype,rank)==(10,1) and n>len(b'prjdatpsana') and wire.cstring(raw[p:p+n])=='xpp101605526', 'Wrong source runinfo')
        out[p:p+n] = b'prjdatpsana'.ljust(n,b'\0')
        p,n,dtype,rank = fields['runnum']
        wire.require((n,dtype,rank)==(4,2,0) and struct.unpack_from('<I',raw,p)[0]==311, 'Wrong source run')
        struct.pack_into('<I',out,p,RUN)
    if wire.service(raw) == 12 and offset is not None:
        (p,_),(q,_) = wire.smd_offsets(raw,schemas)
        struct.pack_into('<Q',out,p,offset)
        struct.pack_into('<Q',out,q,size)
    wire.require(len(out)==len(raw), 'Datagram size changed')
    return bytes(out)


def payload_hash(data):
    return wire.sha(data.encode() if isinstance(data,str) else memoryview(np.ascontiguousarray(data)).cast('B'))


def fingerprint(data):
    wire.require(data is not None, 'Missing detector sample')
    return dict(shape=list(data.shape), dtype=data.dtype.str, sha256=payload_hash(data))
