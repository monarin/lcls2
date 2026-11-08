"""CPU-only 1,000-event source pass and three full-pixel calibration references."""
import argparse
from contextlib import closing
import json
import importlib
from pathlib import Path

from common import ROOT, COUNT, SAMPLES, source_stat, sim_runner, require, file_sha
from psana import DataSource


def main(output):
    require(not output.exists(),'Reference output must be new')
    before=source_stat()
    constants,calibration=sim_runner.stage2.load_constants()
    ds=DataSource(exp='mfx101555026',run=10,dir=str(ROOT),max_events=COUNT,
                  live=False,skip_calib_load='all')
    run=next(ds.runs());det=run.Detector('jungfrau',status=False)
    det.raw._calibconst=constants
    stamps=[];samples={}
    with closing(run.events()) as events:
        for i,event in enumerate(events):
            require(i<COUNT,'CPU source exceeded event cap')
            stamps.append(int(event.timestamp))
            if i in SAMPLES:
                raw=det.raw.raw(event)
                require(raw is not None and raw.shape==(32,512,1024),'Missing/full-shape JF raw')
                calibrated=det.raw.calib(event,cversion=3,cmpars=None).reshape(raw.shape)
                samples[str(i)]=dict(timestamp=stamps[-1],raw_sha256=sim_runner.stage2.digest(raw),
                                    calibrated_sha256=sim_runner.stage2.digest(calibrated),
                                    shape=list(raw.shape),raw_dtype=str(raw.dtype),calibrated_dtype=str(calibrated.dtype))
    require(len(stamps)==COUNT and len(set(stamps))==COUNT and len(samples)==len(SAMPLES),'Incomplete CPU reference')
    pilot=sim_runner.PILOT/'generation.json'
    manifest=json.loads(pilot.read_text())
    require(stamps==[r['timestamp'] for r in manifest['schedule'] if r['service']==12],
            'Real source first 1k differs from sim10 event identities')
    require(source_stat()==before,'Source files changed during reference')
    result=dict(status='pass',source=str(ROOT),events=COUNT,timestamps=stamps,samples=samples,
                source_stat=before,calibration=calibration,pilot_manifest_sha256=file_sha(pilot),
                source_sha256={str(Path(__file__)):file_sha(__file__)},
                cpu_modules={name:dict(path=importlib.import_module(name).__file__,
                                      sha256=file_sha(importlib.import_module(name).__file__))
                             for name in ('psana.detector.jungfrau','psana.detector.UtilsJungfrau')},
                scope='All 1k CPU timestamps; full raw and CPU-v3 calibration hashes at 0/500/999; no full-run file hash',
                cache='CPU pass reads first 1k source events and warms cache; no eviction')
    with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
