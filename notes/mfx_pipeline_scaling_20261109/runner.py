"""MPI whole-loop sample10 cold/warm scaling, all BDs on one GPU."""
import argparse
from contextlib import ExitStack,closing
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch
from mpi4py import MPI
from common import HERE,NOTES,ROOT,require,sha,fixture,condition,cache_state,physical_counters,counter_delta,affinity


def main(args):
    comm=MPI.COMM_WORLD
    require(comm.size==args.bds+2 and 1<=args.bds<=4,'Wrong rank count')
    require(args.bds*args.streams<=4,'Cube bank memory cap: at most four banks')
    chosen=affinity(comm.rank,comm.size)
    from task import Task,Diagnostic,dense_cube
    from instrument import accounting
    import gpu_accept as stage2
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.psexp.run import Run
    manifest,identity,paths=fixture()
    stamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12][:args.events]
    require(len(stamps)==args.events,'Unsupported count')
    constants,calibration=stage2.load_constants()
    recorder=Diagnostic(args.diagnostic)
    task=Task(stamps,args.streams,args.task,recorder)
    dense_cube.adapter.register_reader()
    ds=DataSource(exp='mfx101555026',run=10,dir=str(ROOT),max_events=args.events,live=False,
        skip_calib_load='all',gpu_det='jungfrau',batch_size=args.batch,n_gpu_streams=args.streams,
        gpu_memory_budget_gb=4,gpu_bulk_target_bytes=524288,
        gpu_fn=GpuTask(task,inputs=task.inputs,calibconst=task.calibconst))
    def setup(run):
        run._clear_calibconst();run._calib_const={n:{} for n in run.dsparms.configinfo_dict}
        run._calib_const['jungfrau']=constants;run.dsparms.calibconst=run._calib_const
    with patch.object(Run,'_setup_run_calibconst',setup):run=next(ds.runs())
    role=run.comms.node_type()
    require(role==('smd0' if comm.rank==0 else 'eb' if comm.rank==1 else 'bd'),'Unexpected rank roles')
    report=dict(rank=comm.rank,role=role,host=platform.node(),job=os.getenv('SLURM_JOB_ID'),cpu_affinity=chosen,timestamps=[])
    if role=='bd':
        import cupy as cp
        import kvikio
        import kvikio.defaults as defaults
        require(cp.cuda.runtime.getDeviceCount()==1,'All BDs must see exactly one GPU')
        defaults.set(dict(compat_mode=kvikio.CompatMode.ON,num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False))
        effective={k:defaults.get(k) for k in ('num_threads','task_size','bounce_buffer_size','auto_direct_io_read')}
        require(defaults.get('compat_mode')==kvikio.CompatMode.ON and effective==dict(num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False),'KvikIO settings mismatch')
        report['runtime']=dict(python=sys.version,executable=sys.executable,cupy=cp.__version__,kvikio=kvikio.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            device_pci=cp.cuda.runtime.deviceGetPCIBusId(0),device_mask=os.getenv('CUDA_VISIBLE_DEVICES'),
            mpi=MPI.Get_library_version(),effective=effective,compat_mode='ON')
        modules=('psana','psana.gpu.gpu_events','psana.gpu.gpu_task_batch','psana.gpu.gpu_kvikio_read','psana.gpu.gpu_stream','psana.psexp.event_manager')
        report['runtime_files']={n:dict(path=importlib.import_module(n).__file__,sha256=sha(importlib.import_module(n).__file__)) for n in modules}
        if args.task=='full':task.prepare()
        cp.cuda.Device().synchronize()
        free,total=cp.cuda.runtime.memGetInfo()
        state=args.bds*task.required_state_bytes if args.task=='full' else 0
        require(state+(8<<30)<=free,'Aggregate banks plus reserve exceed free VRAM')
        report['memory_before']=dict(free=free,total=total,aggregate_state_bytes=state)
    comm.Barrier()
    observation=None
    if comm.rank==0:
        idle0=physical_counters();time.sleep(1);idle1=physical_counters()
        observation=dict(idle=dict(before=idle0,after=idle1,delta=counter_delta(idle0,idle1)))
        observation['conditioning']=condition(paths,args.cache)
        observation['nic_before']=physical_counters()
        loop_start=time.perf_counter()
    comm.Barrier()
    rank_start=time.perf_counter()
    with ExitStack() as stack:
        io=None
        if role=='bd':
            io=stack.enter_context(accounting());stack.enter_context(recorder.framework())
        with closing(run.events()) as events:
            for event in events:report['timestamps'].append(int(event.timestamp))
        if role=='bd':task.join();cp.cuda.Device().synchronize()
    report['iteration_seconds']=time.perf_counter()-rank_start
    comm.Barrier()
    if comm.rank==0:
        loop_end=time.perf_counter()
        observation['nic_after']=physical_counters()
        # NIC traffic includes all detectors, metadata, service traffic/readahead.
        observation['nic_delta']=counter_delta(observation['nic_before'],observation['nic_after'])
        observation['cache_after']=cache_state(paths)
        observation['loop_seconds']=loop_end-loop_start
    if role=='bd':
        report['io']=io
        require(io['pressure_drains']==0,'Admission drains confound comparison')
        require(io['requests']==5*len(report['timestamps']) and io['completed_bytes']==3358200*len(report['timestamps']),'JF input accounting mismatch')
        report['checks']=task.validate_local(report['timestamps']);report['callbacks']=task.callbacks
        report['profile']=recorder.report() if args.diagnostic else None
        free,total=cp.cuda.runtime.memGetInfo()
        report['memory_after']=dict(free=free,total=total,pool_used=cp.get_default_memory_pool().used_bytes(),pool_total=cp.get_default_memory_pool().total_bytes())
        task.close()
    report['status']='pass'
    rows=comm.gather(report,root=0)
    if comm.rank==0:
        bds=[r for r in rows if r['role']=='bd'];delivered=[t for r in bds for t in r['timestamps']]
        require(sorted(delivered)==stamps and len(delivered)==len(set(delivered)),'Global event partition mismatch')
        require(all(not r['timestamps'] for r in rows if r['role']!='bd'),'Helper delivered events')
        require(len({r['runtime']['device_pci'] for r in bds})==1,'BDs not sharing one GPU')
        require(len({r['runtime']['device_mask'] for r in bds})==1,'GPU masks differ')
        requested=sum(r['io']['completed_bytes'] for r in bds)
        require(requested==3358200*args.events,'Global JF bytes')
        for name,v in observation['nic_delta'].items():v['rx_to_jf_requested_bytes']=v['rx_bytes']/requested
        source=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'run.sbatch']
        source+=list((NOTES/'mfx_expand10_performance_20261106').glob('*.py'))
        source += [Path(dense_cube.__file__),dense_cube.REFERENCE_FILE,Path(dense_cube.adapter.__file__),dense_cube.adapter.CALIBRATION_FILE,
                   NOTES/'mfx_kvikio_read_20261108/network_watch.py',NOTES/'mfx_kvikio_read_20261108/plan_reads.py']
        result=dict(status='pass',settings=vars(args)|{'output':str(args.output)},identity=identity,calibration=calibration,
            rows=rows,measurement=observation,events=args.events,jf_bytes=requested,
            events_per_second=args.events/observation['loop_seconds'],jf_gib_s=requested/(1<<30)/observation['loop_seconds'],
            source_sha256={str(p):sha(p) for p in source},
            timing='Whole MPI loop from before start barrier to after all ranks complete event delivery and BD GPU joins. Setup, explicit cache conditioning, NIC endpoint calls, validation and cube export excluded. Includes GPU lazy allocation/calibration initialization. No per-event D2H or cross-BD cube merge.')
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(output=str(args.output),status='pass',events_per_second=result['events_per_second'],jf_gib_s=result['jf_gib_s'])),flush=True)
    comm.Barrier()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bds',type=int,required=True);p.add_argument('--streams',type=int,choices=(1,2),required=True)
    p.add_argument('--task',choices=('full','parse'),default='full');p.add_argument('--cache',choices=('cold','warm'),required=True)
    p.add_argument('--events',type=int,default=10000);p.add_argument('--batch',type=int,default=7);p.add_argument('--diagnostic',action='store_true')
    try:main(p.parse_args())
    except BaseException:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
