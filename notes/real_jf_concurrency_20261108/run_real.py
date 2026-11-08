"""One fresh-process 1,000-event real-JF calibration/cube case."""
import argparse
from contextlib import closing
import importlib
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

from common import ROOT, COUNT, SAMPLES, HERE, PREP, source_stat, sim_runner, Recorder, admission_counts, require, file_sha
from cube import Cube
from psana import DataSource
from psana.gpu import GpuTask
from psana.psexp.run import RunSerial


def run(args):
    import cupy as cp
    import kvikio
    require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use frozen runtime')
    reference=json.loads(args.reference.read_text())
    require(reference['status']=='pass' and reference['events']==COUNT,'Invalid reference')
    require(reference['source_stat']==source_stat(),'Source file stat changed')
    stamps=reference['timestamps']
    require(len(stamps)==COUNT and len(set(stamps))==COUNT,'Invalid event identities')
    args.output.mkdir(mode=0o770,exist_ok=False)
    report=dict(status='running',host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),events=COUNT,
                streams=args.streams,bins=100,batch_size=7,bulk_target_bytes=args.target,
                framework_budget_gib=4,source=str(ROOT),reference_sha256=file_sha(args.reference),
                validation_mode=args.validate,delivered_events=0)
    task=None
    memory=sim_runner.measure.Memory()
    def save():
        tmp=args.output/'report.json.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'report.json')
    save()
    try:
        memory.sample('before_setup')
        report['runtime']=dict(python=sim_runner.sys.version,cupy=cp.__version__,kvikio=kvikio.__version__,
                               gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                               cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                               environment={k:os.environ.get(k) for k in ('PS_PARALLEL','PS_FAKESTEP_FLAG','PS_SMD_N_EVENTS',
                                            'PS_SMD_CHUNKSIZE','KVIKIO_COMPAT_MODE','KVIKIO_NTHREADS','CUDA_VISIBLE_DEVICES')})
        modules=('psana','psana.gpu.gpu_events','psana.gpu.gpu_task_batch','psana.gpu.gpu_kvikio_read',
                 'psana.gpu.gpu_stream','psana.gpu.gpudgram.batch','psana.gpu.gpu_input','psana.gpu.gpu_detector',
                 'psana.psexp.event_manager')
        report['runtime_files']={name:dict(path=importlib.import_module(name).__file__,sha256=file_sha(importlib.import_module(name).__file__)) for name in modules}
        dense=sim_runner.dense_cube
        paths=list(HERE.glob('*.py'))+list(PREP.glob('*.py'))+[HERE/'run.sbatch',
               dense.REFERENCE_FILE,Path(dense.__file__),dense.adapter.CALIBRATION_FILE,
               dense.STAGE2/'adapter.py',dense.STAGE2/'gpu_accept.py',dense.STAGE2/'runtime.sh',
               Path(sim_runner.measure.__file__),Path(sim_runner.stage2.fixture.__file__),
               Path(sim_runner.stage2.verify.__file__),Path(importlib.import_module('xtc_fixture').__file__)]
        report['code_sha256']={str(p):file_sha(p) for p in paths}
        constants,report['calibration']=sim_runner.stage2.load_constants()
        require(report['calibration']==reference['calibration'],'CPU/GPU calibration mismatch')
        task=Cube(stamps,warmup=100,recorder=Recorder(False),nbins=100,active_bins=99,
                  max_streams=args.streams,max_state_bytes=27<<30,reference=reference if args.validate else None)
        free,total=cp.cuda.runtime.memGetInfo()
        require(task.required_state_bytes+(8<<30)<=free,'Insufficient cube/device headroom')
        report['admission']=dict(state_bytes=task.required_state_bytes,nonstate_reserve_bytes=8<<30,
                                 free_device_bytes=free,total_device_bytes=total)
        task.prepare()
        ds=DataSource(exp='mfx101555026',run=10,dir=str(ROOT),max_events=COUNT,live=False,
                      skip_calib_load='all',gpu_det='jungfrau',batch_size=7,n_gpu_streams=args.streams,
                      gpu_memory_budget_gb=4,gpu_bulk_target_bytes=args.target,
                      gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst))
        def frozen_setup(run):
            require(run.dsparms is ds.dsparms and run.expt=='mfx101555026' and run.runnum==10,'Unexpected run')
            run._calib_const={'jungfrau':constants};run.dsparms.calibconst=run._calib_const
        with patch.object(RunSerial,'_setup_run_calibconst',frozen_setup):gpu_run=next(ds.runs())
        memory.sample('before_iteration')
        start=time.perf_counter()
        with admission_counts(task.window) as pressure:
            with closing(gpu_run.events()) as events:
                for event in events:
                    i=report['delivered_events']
                    require(i<COUNT and int(event.timestamp)==stamps[i],'CPU/GPU timestamp mismatch or event cap exceeded')
                    report['delivered_events']=i+1
            task.join();report['window']=task.window.finish()
        report['iteration_completed_seconds']=time.perf_counter()-start
        report['admission_pressure_trims']=pressure
        require(report['delivered_events']==COUNT,'Incomplete GPU events')
        report['checks']=task.check_counts()
        report['callbacks']=task.callbacks
        report['bank_events']=[len(bank.events) for bank in task.banks]
        report['pixel_checks']=task.checked
        if args.validate:require(sorted(task.checked)==list(SAMPLES),'Missing pixel acceptance samples')
        report['status']='pass'
        report['timing_contract']='Same rounded warmup/completed-tail contract as sim10 runner. Full-pixel D2H only in validation_mode; its rate is not a throughput result. CPU reference pass warmed cache; no GDS or sustained-rate claim.'
        memory.sample('after_completion')
    except BaseException as exc:
        report.update(status='fail',error=repr(exc));raise
    finally:
        try:
            if task is not None:task.close()
        except BaseException as exc:
            report.update(status='fail',cleanup_error=repr(exc));raise
        finally:
            memory.sample('after_cleanup');report['memory']=memory.summary();save()
    print(json.dumps({k:report[k] for k in ('status','streams','bulk_target_bytes','validation_mode','delivered_events','admission_pressure_trims','window')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--streams',type=int,choices=(1,2),default=2)
    p.add_argument('--target',type=int,choices=(524288,8388608),default=524288)
    p.add_argument('--validate',action='store_true')
    run(p.parse_args())
