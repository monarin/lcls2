"""Offline PR19/20/21 checks against the pinned candidate; only emulated memory."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from collections import Counter

HERE = Path(__file__).parent

def extract(file, name, env):
    tree = ast.parse((HERE / file).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(HERE / file), 'exec'), env)
    return env[name]

class Reg:
    def __init__(self, value=0):
        self.value = value
    def set(self, value):
        self.value = value
    def get(self):
        return self.value

def check_train(failures, prom=True, lane_failure=False):
    attempts = Counter()
    flashed = []
    def frame(self, adc, retry):
        attempts[adc] += 1
        return -1 if not lane_failure and attempts[adc] <= failures.get(adc, 0) else 20
    def lane(self, adc, lane, retry):
        return -1 if lane_failure and lane == 7 and attempts[adc] <= failures.get(adc, 0) else 30
    self = NS(_promWrEn=prom, retries=5, allDelays=[-1]*90,
              SystemRegs=NS(enable=Reg(), AdcBypass=Reg()),
              Ad9249Tester=NS(enable=Reg()),
              Ad9249Readout=[NS(enable=Reg(), FrameDelay=Reg(), ChannelDelay=[Reg() for _ in range(8)]) for _ in range(10)],
              Ad9249Config=[NS(enable=Reg()) for _ in range(10)],
              resetAdc=lambda *a: None, trainFrameAdc=frame, trainDataLaneAdc=lane,
              flashAdcDelays=lambda s: flashed.append(list(s.allDelays)))
    env = dict(self=self, time=NS(sleep=lambda t: None), click=NS(secho=lambda *a, **k: None))
    command = extract('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/Top.py', 'AdcTrain', env)
    with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
        previous = os.getcwd()
        try:
            os.chdir(temp)
            command()
            assert len(Path('ePixQuadAdcTrainingData.txt').read_text().splitlines()) == 90
        finally:
            os.chdir(previous)
    assert bool(flashed) == (prom and all(n < 4 for n in failures.values()))
    assert attempts == Counter({i: min(failures.get(i, 0)+1, 4) for i in range(10)})
    assert self.SystemRegs.AdcBypass.value is False
    assert self.Ad9249Tester.enable.value is False

for failures, prom, lane_failure in [({},True,False), ({},False,False), ({0:4},True,False), ({9:4},True,False), ({0:3,1:3,9:3},True,False), ({0:4,1:3},True,False), ({5:4},True,True), ({5:3},True,True)]:
    check_train(failures, prom, lane_failure)
print('PR20: 8 success/failure/retry scenarios passed; PROM guard, saved diagnostics, and cleanup verified')

class Clock:
    def __init__(self): self.t = 0
    def monotonic(self):
        self.t += .1
        return self.t
    def sleep(self, t): self.t += t

for passed, failed, expected in [(True,False,True), (False,True,False), (False,False,False)]:
    clock = Clock()
    env = dict(time=clock)
    wait = extract('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/Top.py', 'waitAdcTest', env)
    self = NS(testTmo=1, Ad9249Tester=NS(TestPassed=Reg(passed), TestFailed=Reg(failed)))
    with contextlib.redirect_stdout(io.StringIO()):
        assert wait(self) is expected
    assert clock.t <= 1.3

# Use the actual helper and call sites, with stuck completion flags and an accelerated clock.
clock = Clock()
env = dict(time=clock)
wait = extract('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/Top.py', 'waitAdcTest', env)
self = NS(testTmo=1, waitAdcTest=wait, resetAdc=lambda *a:None,
          Ad9249Tester=NS(**{n:Reg() for n in ['TestPassed','TestFailed','TestDataMask','TestPattern','TestSamples','TestTimeout','TestChannel','TestRequest']}),
          Ad9249Config=[NS(OutputTestMode=Reg(), UserPatt1Lsb=Reg(), UserPatt1Msb=Reg())],
          Ad9249Readout=[NS(ChannelDelay=[Reg() for _ in range(8)], LostLockCountReset=lambda:None, LostLockCount=Reg(), Locked=Reg(1))])
with contextlib.redirect_stdout(io.StringIO()):
    assert extract('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/Top.py','testAdc',env)(self,0,0) == -1
    assert extract('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/Top.py','trainDataLaneAdc',env)(self,0,0,1) == -1
assert self.Ad9249Config[0].OutputTestMode.value == 0
print('PR21: pass/fail/timeout and both real call sites passed with fake registers/clock')

import pyrogue as pr
import rogue.interfaces.memory as rim
import pyrogue.interfaces.simulation as sim
spec = importlib.util.spec_from_file_location('review_systemregs', HERE/'/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1/epix-quad-1kfps/software/python/ePixQuad/SystemRegs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class AliasedMemory(sim.MemEmulate):
    def __init__(self):
        super().__init__()
        self.mask = 0xFFFF
        self.mask_writes = []
    def _doTransaction(self, transaction):
        address, size = transaction.address(), transaction.size()
        if address <= 0x28 and address+size >= 0x2C:
            if transaction.type() in (rim.Write, rim.Post):
                data = bytearray(size)
                transaction.getData(data, 0)
                value = int.from_bytes(data[0x28-address:0x2C-address], 'little')
                self.mask_writes.append(value)
                if value >> 16 == 0xAAAA:
                    self.mask = value & 0xFFFF
            # Emulate the distinct readback view, including the cleared key.
            if transaction.type() not in (rim.Write, rim.Post):
                for i, b in enumerate(self.mask.to_bytes(4,'little')):
                    self._data[0x28+i] = b
        super()._doTransaction(transaction)

memory = AliasedMemory()
# Device nodes must be installed before Root.start().
root = pr.Root(pollEn=False, initRead=False)
regs = module.SystemRegs(memBase=memory)
root.add(regs)
with root:
    assert regs.AsicMask.get() == 0xFFFF
    root.WriteAll()
    assert not memory.mask_writes
    regs.AsicMask.set(0xAAAA000F)
    assert memory.mask_writes == [0xAAAA000F]
    assert memory.mask == 0xF
    print('PR19 CAVEAT reproduced: direct RO AsicMask.set issues a keyed hardware write')
    regs.AsicMaskReg.set(0xAAAA000F)
    assert regs.AsicMask.get() == 0xF
    regs.AsicMaskReg.set(0x1234)
    assert regs.AsicMask.get() == 0xF
    root.ReadAll()
print('PR19: real Rogue v6.1.3 SystemRegs builds; WriteAll skips mask, keyed WO write/readback succeeds')
