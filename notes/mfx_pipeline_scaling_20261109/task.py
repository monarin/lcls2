"""MPI-local ownership around unchanged full cube kernels; optional parse control."""
from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
from common import NOTES, require
import sys
sys.path.insert(0,str(NOTES/'mfx_expand10_performance_20261106'))
from perf_cube import Cube as OriginalCube, Capture, dense_cube
from timing import Recorder


class Diagnostic(Recorder):
    def __init__(self,enabled):
        super().__init__(enabled)
        self.stack=[];self.totals=defaultdict(lambda:dict(calls=0,wall_inclusive=0.,wall_exclusive=0.,thread_exclusive=0.))
        self.phase=lambda:'whole_loop'
    @contextmanager
    def cpu(self,name):
        if not self.enabled:
            yield;return
        frame=[0.,0.];self.stack.append(frame)
        start=time.perf_counter();cpu=time.thread_time()
        try:yield
        finally:
            wall=time.perf_counter()-start;thread=time.thread_time()-cpu
            require(self.stack.pop() is frame,'Invalid instrumentation nesting')
            r=self.totals[name];r['calls']+=1;r['wall_inclusive']+=wall
            r['wall_exclusive']+=wall-frame[0];r['thread_exclusive']+=thread-frame[1]
            if self.stack:self.stack[-1][0]+=wall;self.stack[-1][1]+=thread
    def report(self):
        base=super().summary() if self.enabled else None
        return dict(cpu=dict(self.totals),cuda=base['cuda'] if base else [],
            interpretation='Main-thread exclusive wall and thread CPU seconds partition nested hooks. CUDA intervals may overlap and include submission gaps; not a GPU busy-time union.')


class Task(OriginalCube):
    def __init__(self,stamps,streams,kind,recorder):
        super().__init__(stamps,warmup=1,recorder=recorder,nbins=100,active_bins=99,max_streams=streams,max_state_bytes=27*(1<<30))
        self.kind=kind;self.recorder.phase=lambda:'whole_loop'
        if kind=='parse':self.calibconst=()
    def __call__(self,batch,stream):
        self.recorder.callback=len(self.callbacks)
        ts,_=self._identify(batch.timestamps)
        if self.kind=='parse':
            with self.recorder.cpu('descriptor_only_task'):
                field=batch.field('jungfrau','sim10','seed')
                require(tuple(field.segment_ids)==tuple(range(32)),'Wrong panel descriptors')
                require(field.rows.shape[:2]==(batch.size,32),'Wrong descriptor shape')
                self.seen.update(ts)
        else:
            capture=Capture(batch)
            with self.recorder.cpu('expand_calibrate_call'):self.sample10(capture,stream)
            with self.recorder.cpu('cube_bins_upload_accumulate_call'):
                self.add(capture.arrays['jungfrau.calib'],capture.arrays['sim10.errors'],batch.timestamps,stream,batch.keepalive,batch.step_generation)
        self.callbacks.append(dict(size=batch.size,stream=int(stream.ptr),timestamps=list(ts)))
        self.recorder.callback=None
    def validate_local(self,delivered):
        require(len(delivered)==len(set(delivered)) and self.seen==set(delivered),'Task/delivery mismatch')
        require(delivered==sorted(delivered),'Rank event order')
        require([t for c in self.callbacks for t in c['timestamps']]==delivered,'Callback/delivery ordering mismatch')
        counts=np.zeros(100,np.uint64);banks=[]
        if self.kind=='full':
            require(len(self.banks)==self.max_streams,'Execution stream not exercised')
            for bank in self.banks:
                got=bank.counts.get();expected=np.bincount(np.asarray(bank.events,np.int64)%99,minlength=100)
                require(np.array_equal(got,expected),'GPU acceptance or bin error')
                counts+=got;banks.append(dict(events=len(bank.events),counts=got.tolist()))
            expected=np.bincount([self.ordinal[t]%99 for t in delivered],minlength=100)
            require(np.array_equal(counts,expected),'Wrong aggregate local bins')
        return dict(status='pass',events=len(delivered),counts=counts.tolist(),banks=banks)
