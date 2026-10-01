"""Full expert-config regression, offline: real functions and saved TEST map."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
REPLAY = runpy.run_path(str(HERE.parent/'gain-writer-diagnosis/replay.py'))
BASELINE = json.loads((HERE.parent/'gain-writer-diagnosis/replay.json').read_text())['results']


class OrderingTests(unittest.TestCase):
    def replay(self, standalone=False, write_map=True):
        cfg = json.loads(REPLAY['CONFIG'].read_text())
        camera = REPLAY['Recorder'](cfg)
        env = REPLAY['functions']()
        with patch.dict(os.environ, EPIXQUAD_STANDALONE_GAINMAP=str(REPLAY['MASK']) if standalone else ''), contextlib.redirect_stdout(io.StringIO()):
            env['config_expert'](camera.base, copy.deepcopy(cfg), writePixelMap=write_map)
        expected = np.where(np.load(REPLAY['MASK'], allow_pickle=False), 8, 12)
        return camera.result(expected)

    def test_full_daq_configure_writes_with_isen_enabled(self):
        result = self.replay()
        self.assertEqual(result['writes'], {'WriteMatrixData:IsEn=1':16, 'WritePixelData:IsEn=1':22169})
        self.assertEqual(result['isen_history'], [[1,0,1,0]]*16)
        self.assertEqual(result['ideal_map_mismatches'], 0)
        self.assertEqual(result['map_register_trace_sha256'], BASELINE['original_daq_configure']['map_register_trace_sha256'])

    def test_standalone_port_sequence_unchanged(self):
        self.assertEqual(self.replay(standalone=True), BASELINE['standalone_adapter_configure'])

    def test_no_map_update_does_not_enter_programming_again(self):
        result = self.replay(write_map=False)
        self.assertEqual(result['writes'], {})
        self.assertEqual(result['isen_history'], [[1,0,0]]*16)


if __name__ == '__main__':
    unittest.main()
