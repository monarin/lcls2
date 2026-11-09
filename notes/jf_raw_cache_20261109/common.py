"""Strict owner-scoped cache gates and fixed run387 provenance."""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc')
REF=json.loads((HERE/'reference.json').read_text())
NAMES=tuple(REF['10000']['stage_bytes'])
sys.path.insert(0,str(HERE.parent/'mfx_kvikio_read_20261108'))
from network_watch import physical_counters,counter_delta
from plan_reads import run_queue

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def digest(array):
    a=np.array(array,copy=True,order='C')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest(),nonzero=int(np.count_nonzero(a)))

def ts_hash(stamps):return hashlib.sha256(struct.pack('<%dQ'%len(stamps),*stamps)).hexdigest()

def paths(directory):
    directory=Path(directory).resolve()
    local=os.environ.get('JF_LOCAL_INPUT')
    require(directory==SOURCE.resolve() or (local and directory==Path(local).resolve()),'Foreign input root')
    result=[directory/n for n in NAMES]
    for p in result:
        st=p.stat();require(p.resolve().parent==directory and st.st_uid==os.geteuid(),'Foreign cache file')
        require(st.st_size==REF['10000']['stage_bytes'][p.name],'Input size changed')
    return result

def residency(directory,evict=False):
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];libc.mincore.restype=ctypes.c_int
    rows=[]
    for p in paths(directory):
        fd=os.open(p,os.O_RDONLY)
        try:
            n=os.fstat(fd).st_size;pages=(n+mmap.PAGESIZE-1)//mmap.PAGESIZE
            def count():
                with mmap.mmap(fd,n,access=mmap.ACCESS_COPY) as m:
                    address=ctypes.addressof(ctypes.c_char.from_buffer(m));v=np.empty(pages,np.uint8)
                    if libc.mincore(address,n,v.ctypes.data):raise OSError(ctypes.get_errno(),'mincore')
                    return int(np.count_nonzero(v&1))
            before=after=count()
            if evict:
                os.fsync(fd)
                for attempt in range(20):
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);after=count()
                    if after==0:break
                    time.sleep(.1)
                require(after==0,'Failed cold gate '+str(p))
            rows.append(dict(path=str(p),bytes=n,pages=pages,before=before,resident_pages=after))
        finally:os.close(fd)
    return dict(files=rows,pages=sum(x['pages'] for x in rows),resident_pages=sum(x['resident_pages'] for x in rows))

def condition(directory,kind):
    start=time.perf_counter()
    if kind=='cold':residency(directory,True)
    elif kind=='warm':
        from concurrent.futures import ThreadPoolExecutor
        def preload(p):
            with p.open('rb',buffering=0) as f:
                while f.read(16<<20):pass
        with ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(preload,paths(directory)))
    else:raise ValueError('Invalid cache kind')
    cache=residency(directory)
    require(all(x['resident_pages']==0 if kind=='cold' else x['resident_pages']/x['pages']>=.999 for x in cache['files']),'Cache gate failed')
    return dict(kind=kind,seconds=time.perf_counter()-start,cache=cache)

def begin(directory,kind):
    a=physical_counters();time.sleep(1);b=physical_counters()
    return dict(idle=dict(before=a,after=b,delta=counter_delta(a,b)),conditioning=condition(directory,kind),nic_before=physical_counters())

def end(record,directory,seconds):
    record['nic_after']=physical_counters();record['nic_delta']=counter_delta(record['nic_before'],record['nic_after'])
    record['cache_after']=residency(directory);record['loop_seconds']=seconds
    require(all(x['seconds']>=seconds for x in record['nic_delta'].values()),'NIC bracket too short')

def requests(directory,events=10000):
    from psana import dgram
    rows=[];reference=None
    for i,p in enumerate(paths(directory)):
        smd=p.parent/'smalldata'/p.name.replace('.xtc2','.smd.xtc2')
        require(sha(smd)==REF['10000']['smd_hashes'][smd.name],'SMD hash differs')
        fd=os.open(smd,os.O_RDONLY);part=[];stamps=[]
        try:
            cfg=dgram.Dgram(file_descriptor=fd)
            while len(part)<events:
                d=dgram.Dgram(config=cfg)
                if d.service()!=12:continue
                info=d.smdinfo[0].offsetAlg;offset=int(info.intOffset);size=int(info.intDgramSize)
                require(offset>=0 and size>1<<20 and offset+size<=p.stat().st_size,'Bad SMD range')
                require(not part or offset>=part[-1][1]+part[-1][2],'Overlapping ranges')
                part.append((i,offset,size));stamps.append(int(d.timestamp()))
        finally:os.close(fd)
        require(ts_hash(stamps)==REF[str(events)]['timestamp_sha256'],'Wrong timestamps')
        if reference is None:reference=stamps
        require(stamps==reference,'Stream identity mismatch');rows.append(part)
    plan=[r for group in zip(*rows) for r in group]
    require(sum(r[2] for r in plan)==REF[str(events)]['payload_bytes'],'Wrong payload')
    return plan,reference

def sources():
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+list(HERE.glob('*.sbatch'))
    files += [HERE.parent/'mfx_kvikio_read_20261108/network_watch.py',HERE.parent/'mfx_kvikio_read_20261108/plan_reads.py']
    return {str(p):sha(p) for p in files}
