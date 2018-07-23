# NOTE:
# This example only works with psana2-python2.7 environment

import os
from psana import DataSource

def filter(evt):
        return True

os.environ['PS_CALIB_DIR'] = "/reg/d/psdm/cxi/cxid9114/scratch/mona/l2/psana-nersc/demo18/input"
xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona/cxid9114"
ds = DataSource('exp=cxid9114:run=96:dir=%s'%(xtc_dir), filter=filter)
det = None
if ds.nodetype in ("bd", "smd0"):
    det = ds.Detector("DsdCsPad")

for run in ds.runs():
    for evt in run.events():
        if det:
            raw = det.raw(evt)
            ped = det.pedestals(run)
            gain_mask = det.gain_mask(run, gain=6.85)
            print(raw.shape, ped.shape, gain_mask.shape)
