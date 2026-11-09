"""Run the original serial benchmark, adding cache/NIC brackets outside its loop."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import NOTES,affinity,fixture,condition,cache_state,physical_counters,counter_delta,sha
from instrument import accounting
sys.path.insert(0,str(NOTES/'mfx_expand10_performance_20261106'))
import runner as serial_runner
import measure

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',choices=('cold','warm'),required=True);args=p.parse_args()
cpus=affinity(2,3)
_,identity,paths=fixture();observation={};sample=measure.Memory.sample

def measured(self,label):
    if label=='before_iteration':
        observation['conditioning']=condition(paths,args.cache)
        observation['nic_before']=physical_counters()
        observation['envelope_begin']=time.perf_counter()
    if label=='after_completion':
        observation['envelope_end']=time.perf_counter()
        observation['nic_after']=physical_counters()
        observation['nic_delta']=counter_delta(observation['nic_before'],observation['nic_after'],33582000000)
        observation['cache_after']=cache_state(paths)
    return sample(self,label)

original=SimpleNamespace(root=serial_runner.ROOT,output=args.output,mode='throughput',streams=2,batch_size=7,bins=100,warmup=1000,
    framework_budget_gib=4,bulk_target_bytes=524288,reserve_gib=8,state_cap_gib=27,verify_files=False,export=None)
# Align explicit KvikIO settings with the MPI benchmark before handles are opened.
import kvikio
import kvikio.defaults as defaults
defaults.set(dict(compat_mode=kvikio.CompatMode.ON,num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False))
with accounting() as io,patch.object(measure.Memory,'sample',measured):serial_runner.run(original)
native=json.loads((args.output/'report.json').read_text())
if io['completed_bytes']!=33582000000 or io['completed_requests']!=50000 or io['pressure_drains']!=0:raise ValueError('Serial accounting')
record=dict(status='pass',cache=args.cache,cpu_affinity=cpus,measurement=observation,io=io,
    iteration_seconds=native['iteration_completed_seconds'],jf_gib_s=33582000000/(1<<30)/native['iteration_completed_seconds'],
    native_sha256=sha(args.output/'report.json'),source_sha256={str(Path(__file__)):sha(__file__)},identity=identity)
(args.output/'network.json').write_text(json.dumps(record,indent=2)+'\n')
