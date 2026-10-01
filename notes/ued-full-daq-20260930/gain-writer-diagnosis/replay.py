"""Replay actual Configure functions offline; never imports hardware packages.

This is a register-transaction recorder and ideal address model, NOT an ASIC
simulation: IsEn=0 is recorded, but its electrical effects are not invented.
"""
import ast
from collections import Counter, deque
from contextlib import redirect_stdout
import copy
import difflib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from psdaq.configdb.epixquad_layout import RAW_ASIC_LAYOUT, RAW_SHAPE
from psdaq.debugtools.epixquad1kfps import capture_gain_writer as writers
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'psdaq/psdaq/configdb/epixquad1kfps_config.py'
CONFIG = Path('/sdf/home/u/uedopr/daq/gainmap-test-20260930/test-mapfhl.json')
MASK = ROOT / 'notes/ued-commissioning-20260928/map-inputs/expand2.npy'


class Node:
    def __init__(self):
        self.nodes = {}

    def add(self, name, item):
        self.nodes[name] = item
        setattr(self, name, item)
        return item


class Register:
    def __init__(self, value=0, callback=None):
        self.value, self.callback = value, callback

    def get(self, **kwargs):
        return self.value

    def set(self, value):
        self.value = value
        if self.callback:
            self.callback(value)


def translate(name):
    if name == 'TriggerEventBuffer':
        return 'TriggerEventBuffer[0]'
    if name.startswith('Epix10kaSaci'):
        return 'Epix10kaSaci[' + name[12:] + ']'
    if name.startswith('DbgOutSel'):
        return 'DbgOutSel[' + name[9:] + ']'
    return name


def tree(values):
    node = Node()
    for name, value in values.items():
        node.add(translate(name), tree(value) if isinstance(value, dict) else Register())
    return node


class Recorder:
    def __init__(self, cfg):
        self.cam = tree(cfg['expert']['EpixQuad'])
        self.cam.Epix10kaSaci = [self.cam.nodes[f'Epix10kaSaci[{i}]'] for i in range(16)]
        self.cam.SystemRegs.add('TrigEn', Register(True))
        self.cam.SystemRegs.add('AutoTrigEn', Register(False))
        self.cam.RdoutCore.add('RdoutEn', Register(True))
        self.image = np.zeros((16, 178, 192), dtype=np.uint8)
        self.rows, self.cols = [0]*16, [0]*16
        self.writes = Counter()
        self.map_trace_hash = hashlib.sha256()
        self.isen_history = [[] for _ in range(16)]
        for i, saci in enumerate(self.cam.Epix10kaSaci):
            for name in ('enable', 'IsEn', 'PrepareMultiConfig', 'WriteMatrixData',
                         'RowCounter', 'ColCounter', 'WritePixelData'):
                saci.add(name, Register(callback=lambda value, i=i, name=name: self.write(i, name, value)))
        pci = tree({'DevPcie': cfg['expert']['DevPcie']})
        manager = pci.DevPcie.Hsio.TimingRx.TriggerEventManager
        manager.TriggerEventBuffer = [manager.nodes['TriggerEventBuffer[0]']] + [Node() for _ in range(4)]
        for buffer in manager.TriggerEventBuffer:
            buffer.add('MasterEnable', Register(True))
        self.base = dict(cam=self.cam, pci=pci)

    def write(self, i, name, value):
        if name in ('PrepareMultiConfig', 'WriteMatrixData', 'RowCounter', 'ColCounter', 'WritePixelData'):
            self.map_trace_hash.update(f'{i}:{name}:{int(value)}\n'.encode())
        if name == 'IsEn':
            self.isen_history[i].append(int(value))
        elif name == 'RowCounter':
            self.rows[i] = int(value)
        elif name == 'ColCounter':
            self.cols[i] = (0xe, 0xd, 0xb, 0x7).index(value >> 7)*48 + (value & 0x7f)
        elif name in ('WriteMatrixData', 'WritePixelData'):
            isen = int(self.cam.Epix10kaSaci[i].IsEn.value)
            self.writes[f'{name}:IsEn={isen}'] += 1
            if name == 'WriteMatrixData':
                self.image[i].fill(value)
            else:
                self.image[i, self.rows[i], self.cols[i]] = value

    def result(self, expected):
        raw = np.empty(RAW_SHAPE, dtype=np.uint8)
        for s in range(4):
            a = self.image[4*s:4*s+4, :176]
            raw[s, :176, :192] = a[2, ::-1, ::-1]
            raw[s, :176, 192:] = a[1, ::-1, ::-1]
            raw[s, 176:, :192] = a[3]
            raw[s, 176:, 192:] = a[0]
        return dict(writes=dict(self.writes), isen_history=self.isen_history,
                    map_register_trace_sha256=self.map_trace_hash.hexdigest(),
                    ideal_map_mismatches=int(np.count_nonzero(raw != expected)))


def candidate_source():
    original = SOURCE.read_text()
    marker = "    if writePixelMap:\n        if standalone_plan is not None:"
    replacement = """    if writePixelMap:
        # Expert ConfigDB includes the running value IsEn=0. Reassert the
        # programming state after apply_dict, before any matrix/pixel writes.
        for i in asics:
            retry(cbase.Epix10kaSaci[i].IsEn.set, True)
        if standalone_plan is not None:"""
    assert original.count(marker) == 1
    return original.replace(marker, replacement)


def functions(candidate=False):
    module = ast.parse(candidate_source() if candidate else SOURCE.read_text())
    names = {'config_expert', 'apply_dict', 'retry', '_normalize_raw_pixel_map',
             '_raw_pixel_map_shape', '_apply_raw_pixel_map_program', 'mode',
             'get_trigger_buffers', 'epixquad_disable_runtrigger', 'epixquad_enable_runtrigger'}
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(module.body) == len(names)
    env = dict(np=np, os=os, time=time, logging=logging, deque=deque,
               RAW_SHAPE=RAW_SHAPE, RAW_ASIC_LAYOUT=RAW_ASIC_LAYOUT,
               BANK_OFFSETS=gain.BANK_OFFSETS, lane=0, debug_override=None)
    exec(compile(module, str(SOURCE), 'exec'), env)
    return env


def main():
    cfg = json.loads(CONFIG.read_text())
    expected = np.where(np.load(MASK, allow_pickle=False), 8, 12)
    np.testing.assert_array_equal(np.asarray(cfg['user']['pixel_map_raw']).reshape(RAW_SHAPE), expected)
    env = functions()
    results = {}
    for variant in ('original_daq_configure', 'standalone_adapter_configure', 'daq_helper_in_standalone', 'candidate_isen_only'):
        recorder = Recorder(cfg)
        current = functions(candidate=True) if variant == 'candidate_isen_only' else env
        selector = str(MASK) if variant == 'standalone_adapter_configure' else ''
        with patch.dict(os.environ, EPIXQUAD_STANDALONE_GAINMAP=selector), redirect_stdout(io.StringIO()):
            if variant == 'daq_helper_in_standalone':
                selected, _ = gain._selected_pixels_from_map(str(MASK))
                from types import SimpleNamespace
                writers.program(recorder.cam, SimpleNamespace(asics=list(range(16)), mode=gain.GAIN_MODES['mapfhl']),
                                selected, dict(name='daq'), env['_apply_raw_pixel_map_program'])
            else:
                current['config_expert'](recorder.base, copy.deepcopy(cfg))
        results[variant] = recorder.result(expected)
        assert results[variant]['ideal_map_mismatches'] == 0
    assert results['original_daq_configure']['writes'] == {'WriteMatrixData:IsEn=0':16, 'WritePixelData:IsEn=0':22169}
    for name in ('standalone_adapter_configure', 'daq_helper_in_standalone', 'candidate_isen_only'):
        assert results[name]['writes'] == {'WriteMatrixData:IsEn=1':16, 'WritePixelData:IsEn=1':22169}
    assert results['candidate_isen_only']['map_register_trace_sha256'] == results['original_daq_configure']['map_register_trace_sha256']
    patch_path = Path(__file__).with_name('candidate-isen-only.patch')
    patch_path.write_text(''.join(difflib.unified_diff(SOURCE.read_text().splitlines(keepends=True),
                         candidate_source().splitlines(keepends=True),
                         fromfile='a/psdaq/psdaq/configdb/epixquad1kfps_config.py',
                         tofile='b/psdaq/psdaq/configdb/epixquad1kfps_config.py', n=1)))
    print(json.dumps(dict(scope='Offline source-function replay; ideal addressing only, no electrical ASIC model or hardware access',
                         sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE, CONFIG, MASK)},
                         results=results), indent=2))


if __name__ == '__main__':
    main()
