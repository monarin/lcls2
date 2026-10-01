"""Offline checks for targeted addressing and cleanup; no Rogue connection."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).with_name('read-fp-pixels.py')
spec = importlib.util.spec_from_file_location('read_fp_pixels', path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class Variable:
    def __init__(self, value=0):
        self.value = value
        self.path = 'fake'
        self.writes = []

    def get(self, **kwargs):
        return self.value

    def set(self, value):
        self.writes.append(value)
        self.value = value


class Pixel:
    def __init__(self, saci, fail=False):
        self.saci, self.fail, self.addresses = saci, fail, []

    def get(self, read=False):
        assert read and self.saci.IsEn.value
        self.addresses.append((self.saci.RowCounter.value, self.saci.ColCounter.value))
        if self.fail:
            raise RuntimeError('injected read failure')
        return 8

    def set(self, value):
        raise AssertionError('Pixel writes forbidden')


class Tests(unittest.TestCase):
    def make_saci(self, fail=False):
        saci = SimpleNamespace(**{k: Variable() for k in
            ('enable', 'IsEn', 'CmdPrepForRead', 'PrepareMultiConfig', 'RowCounter', 'ColCounter')})
        saci.trbit = Variable(1)
        saci.WritePixelData = Pixel(saci, fail)
        return saci

    def test_exact_targets(self):
        rows, _ = reader.targets(path.parent)
        self.assertEqual(len(rows), 44)
        extra = next(r for r in rows if (r['segment'],r['raw_row'],r['raw_col']) == (1,175,170))
        self.assertEqual((extra['asic'],extra['asic_row'],extra['asic_col'],extra['saci_col']),
                         (6,0,21,0x715))

    def test_read_and_cleanup(self):
        saci, state = self.make_saci(), {}
        rows = [dict(asic_row=0, asic_col=21, saci_col=0x715)]
        reader.read_asic(saci, rows, state)
        self.assertEqual(saci.WritePixelData.addresses, [(0,0x715)]*2)
        self.assertEqual([rows[0]['readback_1'],rows[0]['readback_2']], [8,8])
        self.assertFalse(saci.IsEn.value)
        self.assertFalse(saci.enable.value)
        self.assertEqual(saci.CmdPrepForRead.writes, [0,0])
        self.assertEqual(saci.trbit.writes, [])

    def test_read_failure_restores_asic(self):
        saci, state = self.make_saci(True), {}
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            reader.read_asic(saci, [dict(asic_row=0,saci_col=0x715)], state)
        self.assertFalse(saci.IsEn.value)
        self.assertFalse(saci.enable.value)
        self.assertEqual(saci.CmdPrepForRead.writes, [0,0])
        self.assertEqual(state['cleanup_errors'], [])


if __name__ == '__main__':
    unittest.main()
