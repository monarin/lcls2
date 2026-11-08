"""Real raw -> existing calibration -> existing cube; no expansion or publications."""
import numpy as np
from common import SimCube, dense_cube, sim_runner, require


class Cube(SimCube):
    def __init__(self,*args,reference=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.sample10=None
        self.calibration=dense_cube.adapter.calibration_module.JungfrauCalibration(use_offset=True)
        self.inputs,self.calibconst=self.calibration.inputs,self.calibration.calibconst
        self.reference=reference
        self.checked=[]

    def prepare(self):
        import cupy as cp
        require(self.kernel is None and not self.seen,'Prepare once')
        self.kernel=cp.RawKernel(dense_cube.reference.KERNEL,'accumulate',options=('--std=c++17','--fmad=false'))
        self.accept_kernel=cp.RawKernel(dense_cube.reference.KERNEL,'accept_events',options=('--std=c++17',))
        self.kernel.compile();self.accept_kernel.compile()

    def __call__(self,batch,stream):
        import cupy as cp
        self.window.before(self._warm_join)
        timestamps,bins=self._identify(batch.timestamps)
        require(tuple(batch.segment_ids('jungfrau'))==tuple(range(32)),'Wrong canonical JF panels')
        with stream:
            images,_=self.calibration.calibrate(batch,stream,publish=False)
            present=batch.present('jungfrau.raw')
            errors=cp.empty(present.shape,cp.uint32)
            batch.keepalive(errors)
            cp.subtract(np.uint32(1),present,out=errors)
            self.add(images,errors,batch.timestamps,stream,batch.keepalive,batch.step_generation)
            if self.reference is not None:
                for row,timestamp in enumerate(timestamps):
                    index=self.ordinal[timestamp]
                    expected=self.reference['samples'].get(str(index))
                    if expected is None:continue
                    raw=batch.input('jungfrau.raw')[row].get(stream=stream)
                    calibrated=images[row].get(stream=stream)
                    require(sim_runner.stage2.digest(raw)==expected['raw_sha256'],'Dense real raw differs from CPU')
                    require(sim_runner.stage2.digest(calibrated)==expected['calibrated_sha256'],'Real GPU calibration differs from CPU-v3 bits')
                    self.checked.append(index)
        self.callbacks.append(dict(size=batch.size,stream=int(stream.ptr),phase=self.window.phase))
        self.window.after(batch.size)
