"""Run a bounded capture/rate sequence, or prepare it with no hardware access."""
import argparse
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone

parser=argparse.ArgumentParser()
parser.add_argument('--attempt',type=int,required=True)
parser.add_argument('--mode',choices=('FH','FM','FL','MapFML','MapFHL'),default='FH')
parser.add_argument('--selected-map',type=Path)
parser.add_argument('--frames',type=int,default=50)
parser.add_argument('--gain-writer',choices=('standalone','daq'),default='standalone')
parser.add_argument('--save-config-yaml',action='store_true')
parser.add_argument('--asic-ro-clk-ticks',type=int,choices=(3,6))
rates=parser.add_mutually_exclusive_group()
rates.add_argument('--rate-hz',type=float,default=None)
rates.add_argument('--rate-sequence-hz',type=float,nargs='+')
parser.add_argument('--dry-run',action='store_true',help='Prepare artifacts only; no capture or restoration hardware access')
parser.add_argument('--watchdog-seconds',type=int,default=180)
args=parser.parse_args()
if args.attempt < 1: parser.error('attempt must be positive')
if args.frames < 1 or args.watchdog_seconds < 1: parser.error('frames and watchdog must be positive')
requested_rates=args.rate_sequence_hz or ([args.rate_hz] if args.rate_hz is not None else [10])
if any(not math.isfinite(v) or v <= 0 for v in requested_rates): parser.error('Rates must be positive and finite')
if args.rate_sequence_hz and len(args.rate_sequence_hz)<2: parser.error('Rate sequence requires at least two phases')
if args.mode.startswith('Map') != (args.selected_map is not None):
    parser.error('Map modes require --selected-map; fixed modes must omit it')
if args.selected_map is not None and not args.selected_map.is_file():
    parser.error('Selected map does not exist')
folder=Path(__file__).resolve().parent
checkout=folder.parents[1]
if not args.dry_run and socket.gethostname().split('.')[0]!='drp-ued-cmp003':
    parser.error('Hardware execution is restricted to drp-ued-cmp003')
tools=checkout/'psdaq/psdaq/debugtools/epixquad1kfps'
outdir=folder/('attempt%02d-%s' % (args.attempt,args.mode.lower()))
if outdir.exists() or (folder/('execution-attempt%02d.json' % args.attempt)).exists():
    parser.error('Attempt already exists; choose a new attempt number')
command=[sys.executable,str(tools/'capture_standalone.py'),
         '--profile',str(tools/'profiles/drp-ued-cmp003.json'),
         '--mode',args.mode,'--frames',str(args.frames),'--timeout','30',
         '--gain-writer',args.gain_writer,
         '--label','commissioning-%s-%s-internal-%sHz' % (args.mode,args.gain_writer,'-'.join('%g'%v for v in requested_rates)),
         '--repetition',str(args.attempt),
         '--notes','Production UED commissioning capture; restore camera idle trigger state afterward',
         '--outdir',str(outdir)]
if args.selected_map is not None:
    command += ['--selected-map',str(args.selected_map.resolve()),'--no-default-pixels']
if args.rate_sequence_hz:
    command += ['--rate-sequence-hz']+[str(v) for v in args.rate_sequence_hz]
elif args.rate_hz is not None:
    command += ['--rate-hz',str(args.rate_hz)]
if args.dry_run:
    command += ['--dry-run']
if args.save_config_yaml:
    command += ['--save-config-yaml']
if args.asic_ro_clk_ticks is not None:
    command += ['--asic-ro-clk-ticks',str(args.asic_ro_clk_ticks)]
record={'started_utc':datetime.now(timezone.utc).isoformat(),'command':command,
        'watchdog_seconds':args.watchdog_seconds}
child=None
try:
    child=subprocess.Popen(command)
    try:
        record['capture_exit']=child.wait(timeout=args.watchdog_seconds)
    except subprocess.TimeoutExpired:
        record['watchdog']='%d-second setup/acquisition watchdog expired' % args.watchdog_seconds
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
    if args.dry_run:
        record['restore_skipped']='dry run; no hardware access'
    else:
        # A failed restore must not prevent writing the execution audit record.
        # Remove the previous result so it cannot be mistaken for this attempt.
        (folder/'restore.json').unlink(missing_ok=True)
        try:
            with (folder/'restore.log').open('w') as log:
                restored=subprocess.run([sys.executable,str(folder/'restore_camera_idle.py')],
                                        stdout=log,stderr=subprocess.STDOUT,timeout=45)
            record['restore_exit']=restored.returncode
        except Exception as exc:
            record['restore_error']=repr(exc)
        for name in ('restore.json','restore.log'):
            source=folder/name
            if source.exists():
                (folder/(source.stem+'-attempt%02d' % args.attempt+source.suffix)).write_bytes(source.read_bytes())
            else:
                record.setdefault('restore_missing_artifacts',[]).append(name)
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/('execution-attempt%02d.json' % args.attempt)).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
restore_ok=(record.get('restore_exit')==0 and not record.get('restore_missing_artifacts'))
sys.exit(0 if record.get('capture_exit')==0 and (args.dry_run or restore_ok) else 1)
