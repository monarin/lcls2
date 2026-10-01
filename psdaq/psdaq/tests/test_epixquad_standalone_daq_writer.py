import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from psdaq.debugtools.epixquad1kfps import standalone_daq_writer as adapter
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain


class Variable:
    def __init__(self, name, trace, value=0, fail=False):
        self.name, self.trace, self.value, self.fail = name, trace, value, fail

    def set(self, value):
        if self.fail:
            raise RuntimeError('injected pixel write failure')
        self.trace.append((self.name, value))
        self.value = value

    def get(self, **kwargs):
        return self.value


def camera(fail=False):
    trace, asics = [], []
    for i in range(16):
        asics.append(SimpleNamespace(**{key: Variable(f'a{i}.{key}', trace,
            fail=fail and key == 'WritePixelData') for key in
            ('enable','IsEn','trbit','PrepareMultiConfig','WriteMatrixData',
             'RowCounter','ColCounter','WritePixelData')}))
    return SimpleNamespace(Epix10kaSaci=asics,
        SystemRegs=SimpleNamespace(TrigEn=Variable('TrigEn',trace,True),
                                  AutoTrigEn=Variable('AutoTrigEn',trace,False)),
        RdoutCore=SimpleNamespace(RdoutEn=Variable('RdoutEn',trace,True))), trace


class Tests(unittest.TestCase):
    def setUp(self):
        self.mask = Path(__file__).resolve().parents[3]/'notes/ued-commissioning-20260928/map-inputs/expand2.npy'
        raw = np.load(self.mask)
        self.cfg = dict(user=dict(gain_mode=5,pixel_map_raw=np.where(raw,8,12)),
                        expert=dict(EpixQuad={f'Epix10kaSaci{i}':dict(trbit=1) for i in range(16)}))

    def test_identical_standalone_register_sequence(self):
        plan = adapter.prepare(self.cfg,list(range(16)),self.mask)
        self.assertEqual(len(plan['selected']),22169)
        a, trace_a = camera()
        b, trace_b = camera()
        with contextlib.redirect_stdout(io.StringIO()):
            gain._write_gain_mode(a,gain.GAIN_MODES['mapfhl'],list(range(16)),plan['selected'],[],0)
            adapter.program(b,plan)
        self.assertEqual(trace_a, [row for row in trace_b if row[0].startswith('a')])
        self.assertEqual(sum(name.endswith('.WritePixelData') for name,_ in trace_a),22169)
        self.assertEqual(sum(name.endswith('.WriteMatrixData') for name,_ in trace_a),16)
        self.assertTrue(b.SystemRegs.TrigEn.value and b.RdoutCore.RdoutEn.value)
        self.assertFalse(b.SystemRegs.AutoTrigEn.value)

    def test_reject_metadata_mismatch(self):
        self.cfg['user']['pixel_map_raw'][0,0,0] ^= 4
        with self.assertRaises(AssertionError):
            adapter.prepare(self.cfg,list(range(16)),self.mask)

    def test_reject_trbit_mismatch(self):
        self.cfg['expert']['EpixQuad']['Epix10kaSaci0']['trbit'] = 0
        with self.assertRaisesRegex(ValueError,'trbit'):
            adapter.prepare(self.cfg,list(range(16)),self.mask)

    def test_write_failure_keeps_gates_off_and_cleans_asics(self):
        plan = adapter.prepare(self.cfg,list(range(16)),self.mask)
        c, _ = camera(fail=True)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            adapter.program(c,plan)
        self.assertFalse(c.SystemRegs.TrigEn.value)
        self.assertFalse(c.RdoutCore.RdoutEn.value)
        self.assertTrue(all(not a.IsEn.value and not a.enable.value for a in c.Epix10kaSaci))


if __name__ == '__main__':
    unittest.main()
