"""Fixture-specific cache protocol and physical NIC snapshots."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
NOTES=HERE.parent
sys.path.insert(0,str(NOTES/'mfx_kvikio_read_20261108'))
from network_watch import physical_counters, counter_delta
from plan_reads import require, sha
from fixture_check import ROOT, check_10k


def fixture():
    manifest,identity=check_10k(ROOT,sha)
    paths=[Path(s['files']['bd']['path']) for s in manifest['streams']]
    require(len(set(paths))==10,'Expected ten bigdata files')
    return manifest,identity,paths


def cache_state(paths, evict=False):
    _,_,allowed=fixture()
    require(set(map(Path,paths))==set(allowed),'Cache protocol requires exactly the ten accepted bigdata files')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    libc.mincore.restype=ctypes.c_int
    records=[]
    for path in paths:
        require(path.resolve().parent==ROOT.resolve(),'Foreign cache target')
        fd=os.open(path,os.O_RDONLY)
        try:
            st=os.fstat(fd);require(st.st_uid==os.geteuid(),'Owner required for trustworthy mincore')
            pages=(st.st_size+mmap.PAGESIZE-1)//mmap.PAGESIZE
            def count():
                with mmap.mmap(fd,st.st_size,access=mmap.ACCESS_COPY) as mapping:
                    vector=np.empty(pages,dtype=np.uint8)
                    address=ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                    if libc.mincore(address,st.st_size,vector.ctypes.data):
                        raise OSError(ctypes.get_errno(),'mincore')
                    return int(np.count_nonzero(vector&1))
            before=after=count()
            if evict:
                os.fsync(fd)
                for attempt in range(20):
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);after=count()
                    if after==0:break
                    time.sleep(.1)
                require(after==0,'Failed eviction '+str(path))
            records.append(dict(path=str(path),pages=pages,bytes=st.st_size,before=before,resident_pages=after))
        finally:os.close(fd)
    return dict(files=records,pages=sum(r['pages'] for r in records),resident_pages=sum(r['resident_pages'] for r in records))


def condition(paths,kind):
    begin=time.perf_counter()
    if kind=='cold':cache_state(paths,evict=True)
    elif kind=='warm':
        for path in paths:
            with path.open('rb',buffering=0) as f:
                while f.read(16<<20):pass
    else:raise ValueError('Unknown cache mode')
    result=cache_state(paths)
    fraction=result['resident_pages']/result['pages']
    require(fraction==0 if kind=='cold' else fraction>=.999,'Cache gate failed')
    return dict(kind=kind,seconds=time.perf_counter()-begin,cache=result)


def affinity(rank,size):
    allowed=sorted(os.sched_getaffinity(0));cores=[];seen=set()
    for cpu in allowed:
        top=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key=((top/'physical_package_id').read_text().strip(),(top/'core_id').read_text().strip())
        if key not in seen:cores.append(cpu);seen.add(key)
    require(len(cores)>=2+4*(size-2),'Insufficient distinct physical CPU cores')
    chosen=cores[rank:rank+1] if rank<2 else cores[2+4*(rank-2):2+4*(rank-1)]
    os.sched_setaffinity(0,chosen)
    return chosen
