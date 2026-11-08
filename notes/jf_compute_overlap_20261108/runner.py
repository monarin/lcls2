"""Bounded real-JF pipeline diagnostic with twelve calibration passes per task."""
import argparse
from contextlib import contextmanager, ExitStack
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'real_jf_concurrency_20261108'
sys.path.insert(0,str(PREVIOUS))
from common import sim_runner, ROOT, source_stat, file_sha, require, Recorder
from cube import Cube as RealCube
from mpi4py import MPI
from psana import DataSource
from psana.gpu import GpuTask
from psana.psexp.run import Run

COUNT=64
REPEATS=12


@contextmanager
def span(name):
    import cupy as cp
    cp.cuda.nvtx.RangePush(name)
    try:yield
    finally:cp.cuda.nvtx.RangePop()


class Resident:
    """Owned raw copies and constants survive pipeline drain; never retain a context."""
    def __init__(self,batch):
        self.raw=batch.input('jungfrau.raw').copy()
        self.presence=batch.present('jungfrau.raw').copy()
        self.constants={key:batch.calibconst(det,key) for det,key in (('jungfrau','pedestals'),('jungfrau','pixel_gain'),('jungfrau','pixel_offset'))}
        self.timestamps=tuple(int(t) for t in batch.timestamps)
        self.size=batch.size
        self.owners=[]
    def input(self,name):return self.raw
    def present(self,name):return self.presence
    def segment_ids(self,name):return tuple(range(32))
    def calibconst(self,det,key):return self.constants[key]
    def keepalive(self,*arrays):self.owners.extend(arrays)


class RepeatedCalibration:
    """Replay the fixed unmasked calibration into its same batch-owned output.

    Only this diagnostic uses the example's private kernel cache/ABI. Every
    replay reads original uint16 raw, never the previous calibrated output.
    Cube accumulation remains once per event. No added host synchronization.
    """
    def __init__(self, base):
        self.base = base
        self.sample = None

    def calibrate(self, batch, stream, *, publish=False):
        import numpy as np
        require(not publish and self.base.use_offset and not self.base.status_bits
                and not self.base.stextra_bits, 'Fixed diagnostic calibration ABI only')
        with stream, span('calibration_repeat12_enqueue'):
            out, valid = self.base.calibrate(batch, stream, publish=False)
            raw = batch.input('jungfrau.raw')
            present = batch.present('jungfrau.raw')
            peds, gain, offset = (batch.calibconst('jungfrau', k)
                                 for k in ('pedestals', 'pixel_gain', 'pixel_offset'))
            require(valid is None and len(self.base._kernels) == 1, 'Unexpected kernel variant')
            kernel, = self.base._kernels.values()
            args = (raw, present, peds, gain, offset, raw, out,
                    np.uint64(raw.size), np.uint64(raw.shape[2] * raw.shape[3]),
                    np.uint64(peds.shape[1]), np.uint64(0), raw, np.uint64(0), raw)
            for _ in range(REPEATS - 1):
                kernel((min(65535, (raw.size + 255) // 256),), (256,), args, stream=stream)
            if self.sample is None:
                # Own a copy until validation after the pipeline's explicit join.
                self.sample = out[0].copy()
        return out, valid


class Cube(RealCube):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.resident=[]
        self.calibration=RepeatedCalibration(self.calibration)
    def __call__(self,batch,stream):
        with span('task events=%d stream=%s'%(batch.size,stream.ptr)):
            # Two warmup-only snapshots for the subsequent positive control.
            if len(self.resident)<2:
                with stream,span('save_control_inputs_warmup_only'):
                    self.resident.append(Resident(batch))
            super().__call__(batch,stream)


@contextmanager
def instrumentation(records):
    from psana.gpu.gpu_events import GpuEventManager as Manager
    with ExitStack() as stack:
        original=Manager._split_subbatches
        def split(manager,view):
            with span('admission_split'):
                ranges=original(manager,view)
                costs=manager._event_memory(view)
                records['splits'].append(dict(events=len(costs),sizes=[s._end-s._start for s in ranges],
                    capacity=manager._admission_capacity,depth=manager.event_pool.depth,
                    event_costs=[e.detector_bytes+sum(n for _,n in e.streams) for e in costs],
                    committed=manager._gpu_budget.committed()))
                return ranges
        stack.enter_context(patch.object(Manager,'_split_subbatches',split))
        for method,label in (('_issue_group_reads','read_issue'),('_wait_gpu_read','read_wait'),
                             ('_submit_group_gpu','submit_gather_task'),('_trim_gpu_caches','admission_pressure_trim')):
            base=getattr(Manager,method)
            def wrap(base=base,label=label):
                @functools.wraps(base)
                def call(*args,**kwargs):
                    if label=='admission_pressure_trim':records['pressure_drains']+=1
                    with span(label):return base(*args,**kwargs)
                return call
            stack.enter_context(patch.object(Manager,method,wrap()))
        yield


def resident_control(task):
    import cupy as cp
    import numpy as np
    batches=task.resident
    require(len(batches)==2 and all(b.size==4 for b in batches),'Need two four-event control inputs')
    stamps=[t for b in batches for t in b.timestamps]
    cube=RealCube(stamps,warmup=1,recorder=Recorder(False),nbins=2,active_bins=2,
                  max_streams=2,max_state_bytes=512<<20)
    cube.prepare()
    cube.calibration=task.calibration.base  # Control retains one pass per batch.
    streams=[cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    try:
        for s in streams:
            with s:cube._bank(s)
        cp.cuda.Device().synchronize()  # Preallocate/initialize before control range.
        with span('RESIDENT_CONTROL_8_EVENTS'):
            for b,s in zip(batches,streams):
                with s,span('resident_task events=4'):
                    images,_=cube.calibration.calibrate(b,s,publish=False)
                    errors=cp.empty(b.presence.shape,cp.uint32);b.keepalive(errors)
                    cp.subtract(np.uint32(1),b.presence,out=errors)
                    cube.add(images,errors,b.timestamps,s,b.keepalive)
            cube.join()
        checks=cube.check_counts()
        # Full first raw/calibrated image acceptance after all control GPU work.
        ref=json.loads((PREVIOUS/'reference.json').read_text())['samples']['0']
        raw=batches[0].raw[0].get()
        calibrated=next(a for a in batches[0].owners if a.shape==batches[0].raw.shape and a.dtype==cp.float32)[0].get()
        require(sim_runner.stage2.digest(raw)==ref['raw_sha256'],'Control raw differs from CPU reference')
        require(sim_runner.stage2.digest(calibrated)==ref['calibrated_sha256'],'Control calibration differs from CPU reference')
        return dict(events=8,sizes=[4,4],streams=[int(s.ptr) for s in streams],checks=checks,pixel_sample=0,
                    scope='Same calibration and cube kernels on independent resident inputs; no I/O or dense gather; no performance claim')
    finally:
        cube.close()
        for b in batches:b.owners.clear()


def main(args):
    comm=MPI.COMM_WORLD
    require(comm.size==3,'Exactly SMD0 + EB + one BD required')
    reference=json.loads((PREVIOUS/'reference.json').read_text())
    require(source_stat()==reference['source_stat'],'Original source changed')
    constants,calibration=sim_runner.stage2.load_constants()
    require(calibration==reference['calibration'],'Frozen constants changed')
    task=Cube(reference['timestamps'][:COUNT],warmup=16,recorder=Recorder(False),
              nbins=2,active_bins=2,max_streams=2,max_state_bytes=512<<20)
    ds=DataSource(exp='mfx101555026',run=10,dir=str(ROOT),max_events=COUNT,live=False,
                  skip_calib_load='all',gpu_det='jungfrau',batch_size=16,n_gpu_streams=2,
                  gpu_memory_budget_gb=1.25,gpu_bulk_target_bytes=524288,
                  gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst))
    def frozen_setup(run):
        run._clear_calibconst()
        run._calib_const={name:{} for name in run.dsparms.configinfo_dict}
        run._calib_const['jungfrau']=constants;run.dsparms.calibconst=run._calib_const
    with patch.object(Run,'_setup_run_calibconst',frozen_setup):run=next(ds.runs())
    role=run.comms.node_type()
    require(type(run).__name__=='RunParallel','Did not use MPI RunParallel')
    report=dict(rank=comm.rank,role=role,host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                status='running',timestamps=[],splits=[],pressure_drains=0)
    if role=='bd':
        import cupy as cp
        import kvikio
        task.prepare()
        report['runtime']=dict(python=sys.version,cupy=cp.__version__,kvikio=kvikio.__version__,
            gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),mpi=MPI.Get_library_version(),
            environment={k:os.environ.get(k) for k in ('PS_PARALLEL','PS_EB_NODES','PS_SRV_NODES','PS_SMD_N_EVENTS','KVIKIO_COMPAT_MODE','CUDA_VISIBLE_DEVICES')})
        modules=('psana.gpu.gpu_events','psana.gpu.gpu_stream','psana.gpu.gpu_group_schedule',
                 'psana.gpu.gpu_input_group','psana.gpu.gpu_admission','psana.gpu.gpu_detector','psana.gpu.gpu_task_batch')
        report['runtime_files']={n:dict(path=importlib.import_module(n).__file__,sha256=file_sha(importlib.import_module(n).__file__)) for n in modules}
        report['reference_sha256']=file_sha(PREVIOUS/'reference.json')
        report['source_sha256']={str(p):file_sha(p) for p in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'run.sbatch',PREVIOUS/'cube.py',PREVIOUS/'common.py']}
        report['settings']=dict(events=COUNT,batch_size=16,pool_depth=2,n_gpu_streams=2,target_subbatch=4,
            gpu_memory_budget_gb=1.25,gpu_bulk_target_bytes=524288,bins=2,calibration_repeats=REPEATS,state_bytes=task.required_state_bytes)
    comm.Barrier()
    try:
        with ExitStack() as stack:
            if role=='bd':
                stack.enter_context(span('PIPELINE_64_EVENTS'))
                stack.enter_context(instrumentation(report))
            for event in run.events():report['timestamps'].append(int(event.timestamp))
            if role=='bd':task.join()
        if role=='bd':
            require(report['timestamps']==reference['timestamps'][:COUNT],'Wrong 64 event identities')
            require(len(task.callbacks)==16 and all(c['size']==4 for c in task.callbacks),'Expected sixteen four-event executions')
            require(len(report['splits'])==4 and all(s['sizes']==[4]*4 for s in report['splits']),'Expected four complete EB batches split 4+4+4+4')
            require(report['pressure_drains']==0,'Admission pressure confounds this timeline')
            report['callbacks']=task.callbacks;report['checks']=task.check_counts()
            sample_sha=sim_runner.stage2.digest(task.calibration.sample.get())
            require(sample_sha==reference['samples']['0']['calibrated_sha256'], 'Repeated output differs from CPU-v3 reference')
            require(task.resident[0].timestamps[0]==reference['timestamps'][0], 'Wrong validation sample identity')
            report['repeated_calibration_sample_sha256']=sample_sha
            report['resident_control']=resident_control(task)
            report['bank_events']=[len(b.events) for b in task.banks]
            require(report['bank_events']==[32,32],'Wrong pool use')
            report['source_stat_unchanged']=source_stat()==reference['source_stat']
            require(report['source_stat_unchanged'],'Source changed during trace')
    finally:
        if role=='bd':
            task.close()
    report['status']='pass'
    rows=comm.gather(report,root=0)
    if comm.rank==0:
        require(sorted(r['role'] for r in rows)==['bd','eb','smd0'],'Wrong rank topology')
        require(all(not r['timestamps'] for r in rows if r['role']!='bd'),'Helper delivered events')
        (args.output/'report.json').write_text(json.dumps(dict(status='pass',rows=rows),indent=2)+'\n')
        print('PASS: 64 events, 1 BD/1 GPU, depth 2, four 16-event batches, 16 four-event executions',flush=True)
    comm.Barrier()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    try:main(p.parse_args())
    except BaseException:
        import traceback
        traceback.print_exc();MPI.COMM_WORLD.Abort(1)
