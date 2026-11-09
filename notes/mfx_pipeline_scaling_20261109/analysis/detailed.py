"""One focused diagnostic: partition formerly uninstrumented controller/MPI work."""
import argparse
from contextlib import contextmanager,ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import sha
import task

class Detailed(task.Diagnostic):
    @contextmanager
    def framework(self):
        from psana.gpu.gpu_events import GpuEventManager
        from psana.gpu.gpu_file_epochs import GpuFileEpochs
        from psana.psexp.node import BigDataNode
        from psana.psexp.events import Events
        from psana.psexp.event_manager import EventManager
        from psana.psexp.run import Run
        hooks=[(GpuEventManager,'_process_batch','gpu_batch_controller',True),
            (GpuEventManager,'_split_subbatches','subbatch_planning',False),
            (GpuEventManager,'_issue_group_reads','input_group_issue',False),
            (GpuEventManager,'_submit_group_gpu','input_group_parse_submit',False),
            (GpuEventManager,'_handle_steps','step_handling',True),
            (GpuEventManager,'_setup_gpu_pipeline','gpu_pipeline_initialization',False),
            (GpuEventManager,'_close_resources','gpu_pipeline_close',False),
            (GpuEventManager,'_attach_gpu','attach_gpu_event',False),
            (GpuFileEpochs,'resolve','file_epoch_resolution',False),
            (BigDataNode,'_batch_envelopes','mpi_batch_receive_unpack_request',True),
            (EventManager,'__init__','cpu_eventmanager_initialization',False),
            (Events,'__next__','events_iterator',False),
            (Run,'_materialize_event','run_event_materialization',False)]
        with super().framework(),ExitStack() as stack:
            for cls,attr,name,generator in hooks:
                stack.enter_context(patch.object(cls,attr,self.wrap(name,getattr(cls,attr),generator)))
            yield

task.Diagnostic=Detailed
spec=importlib.util.spec_from_file_location('mpi_scaling_entry',HERE/'runner.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.bds=1;args.streams=2;args.task='full';args.cache='cold';args.events=10000;args.batch=7;args.diagnostic=True
try:
    entry.main(args)
    if entry.MPI.COMM_WORLD.rank==0:
        report=json.loads(args.output.read_text())
        report['additional_instrumentation']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),
            note='Only diagnostic hooks added; source/kernel/controller behavior unchanged. CPU spans exclusive and nested; no throughput claim from this profiled run.')
        args.output.write_text(json.dumps(report,indent=2)+'\n')
except BaseException:
    import traceback;traceback.print_exc();entry.MPI.COMM_WORLD.Abort(1)
