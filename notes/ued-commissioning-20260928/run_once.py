"""Run one bounded 50-frame attempt and restore preflight camera idle settings."""
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone

parser=argparse.ArgumentParser()
parser.add_argument('--attempt',type=int,required=True)
parser.add_argument('--mode',choices=('FH','FM','FL'),default='FH')
args=parser.parse_args()
if args.attempt < 1: parser.error('attempt must be positive')
folder=Path(__file__).resolve().parent
checkout=folder.parents[1]
assert socket.gethostname().split('.')[0]=='drp-ued-cmp003'
tools=checkout/'psdaq/psdaq/debugtools/epixquad1kfps'
outdir=folder/('attempt%02d-%s' % (args.attempt,args.mode.lower()))
if outdir.exists() or (folder/('execution-attempt%02d.json' % args.attempt)).exists():
    parser.error('Attempt already exists; choose a new attempt number')
command=[sys.executable,str(tools/'capture_standalone.py'),
         '--profile',str(tools/'profiles/drp-ued-cmp003.json'),
         '--mode',args.mode,'--frames','50','--timeout','30',
         '--label','commissioning-%s-internal-10Hz' % args.mode,'--repetition',str(args.attempt),
         '--notes','Production UED commissioning capture; restore camera idle trigger state afterward',
         '--outdir',str(outdir)]
record={'started_utc':datetime.now(timezone.utc).isoformat(),'command':command}
child=None
try:
    child=subprocess.Popen(command)
    try:
        record['capture_exit']=child.wait(timeout=180)
    except subprocess.TimeoutExpired:
        record['watchdog']='180-second setup/acquisition watchdog expired'
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        record['capture_exit']=child.returncode
finally:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    with (folder/'restore.log').open('w') as log:
        restored=subprocess.run([sys.executable,str(folder/'restore_camera_idle.py')],
                                stdout=log,stderr=subprocess.STDOUT,timeout=45)
    record['restore_exit']=restored.returncode
    for name in ('restore.json','restore.log'):
        source=folder/name
        (folder/(source.stem+'-attempt%02d' % args.attempt+source.suffix)).write_bytes(source.read_bytes())
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/('execution-attempt%02d.json' % args.attempt)).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if record.get('capture_exit')==0 and record.get('restore_exit')==0 else 1)
