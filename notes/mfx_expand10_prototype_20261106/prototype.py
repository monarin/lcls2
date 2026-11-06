"""Small native GpuTask entrypoint: sample10 expansion -> calibration -> cube.

Reuses the independently reviewed Stage 3A task, not the CPU validation harness.
Use the Stage 2 runtime.sh. No new XTC, database writes or managed lifecycle API.
"""
import argparse
import importlib
from contextlib import closing
import json
import os
import platform
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

NOTES=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(NOTES/'mfx_expand10_stage3a_20261105'))
from cube import Cube, adapter, STAGE2
from gpu_accept import load_constants, preflight
from fixture import file_sha, require
from psana import DataSource
from psana.gpu import GpuTask
from psana.psexp.run import RunSerial

DEFAULT_ROOT=Path('/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261105-stage2-r1/pilot')


def run(root,output):
    require(os.environ.get('PS_PARALLEL')=='none' and os.environ.get('PS_FAKESTEP_FLAG')=='0','Use the recorded Stage 2 runtime')
    output.mkdir(mode=0o700,exist_ok=False)
    report={'status':'running','fixture':str(root),'events':0,'batch_size':7,'n_gpu_streams':2,
            'nbins':30,'active_bins':29,'timing_scope':'One small correctness/prototype run, not a sustained-throughput measurement'}
    task=None; start=time.monotonic()
    try:
        import cupy as cp
        import kvikio
        report['runtime']={'host':platform.node(),'python':sys.version,'executable':sys.executable,
                           'cupy':cp.__version__,'kvikio':kvikio.__version__,
                           'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                           'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(),
                           'KVIKIO_COMPAT_MODE':os.environ.get('KVIKIO_COMPAT_MODE'),
                           'job':os.environ.get('SLURM_JOB_ID')}
        report['runtime_files']={}
        for name in ('psana','psana.gpu.gpu_events','psana.gpu.gpu_task_batch'):
            path=Path(importlib.import_module(name).__file__)
            report['runtime_files'][name]={'path':str(path),'sha256':file_sha(path)}
        manifest=preflight(root)
        summary=json.loads((STAGE2/'validation-summary.json').read_text())
        expected=[v for v in summary['fixtures'].values() if Path(v['generation']['path'])==root/'generation.json']
        require(len(expected)==1 and file_sha(root/'generation.json')==expected[0]['generation']['sha256'],'Use an accepted Stage 2 fixture')
        stamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12]
        constants,report['calibration']=load_constants()
        adapter.register_reader()
        # Host-only declaration. Cube owns one separate bank per execution stream.
        task=Cube(stamps,nbins=30,active_bins=29,max_streams=2)
        ds=DataSource(exp='mfx101555026',run=10,dir=str(root),live=False,
                      skip_calib_load='all',gpu_det='jungfrau',batch_size=7,n_gpu_streams=2,
                      gpu_memory_budget_gb=4,gpu_bulk_target_bytes=8<<20,
                      gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst))
        # Test-only frozen-constant loader; constructor creates GPU manager before
        # runs() yields. Restored immediately; no installed psana files change.
        def frozen_setup(run):
            require(run.dsparms is ds.dsparms,'Unexpected run constructor')
            run._calib_const={'jungfrau':constants};run.dsparms.calibconst=run._calib_const
        with patch.object(RunSerial,'_setup_run_calibconst',frozen_setup):
            run=next(ds.runs())
        with closing(run.events()) as events:
            for i,event in enumerate(events):
                require(i<len(stamps) and int(event.timestamp)==stamps[i],'Event ordering mismatch')
                require(not np.any(event.gpu.get('sim10.errors').on_cpu),'Malformed seed input')
                report['events']+=1
        require(report['events']==len(stamps),'Incomplete run')
        # EOF plus explicit join/export. No per-event image copies or CPU calibration.
        sums,counts=task.finish()
        expected_counts=np.bincount(np.arange(len(stamps))%29,minlength=30)
        require(np.array_equal(counts,expected_counts),'Wrong bin counts')
        require(np.isfinite(sums).all() and not np.any(sums[29]),'Invalid/empty bin values')
        np.save(output/'cube.npy',sums)
        np.save(output/'counts.npy',counts)
        report.update(status='pass',counts=counts.tolist(),cube_shape=list(sums.shape),cube_dtype=str(sums.dtype),
                      bank_event_counts=[len(b.events) for b in task.banks],callback_sizes=task.batch_sizes,
                      pool_peak_bytes=task.pool_peak_bytes,export_count=task.export_count,
                      generation_sha256=file_sha(root/'generation.json'),
                      code_sha256={str(p):file_sha(p) for p in [Path(__file__),NOTES/'mfx_expand10_stage3a_20261105/cube.py',
                          STAGE2/'adapter.py',adapter.CALIBRATION_FILE,STAGE2/'gpu_accept.py',STAGE2/'runtime.sh',
                          Path(__file__).with_name('run.sbatch')]})
    except BaseException as exc:
        report.update(status='fail',error=repr(exc));raise
    finally:
        try:
            if task is not None:task.close()
        except BaseException as exc:
            report.update(status='fail',cleanup_error=repr(exc));raise
        finally:
            report['elapsed_seconds']=time.monotonic()-start
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'events':report['events'],'counts':report['counts'],'output':str(output)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
