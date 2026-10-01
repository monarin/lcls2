"""Opt-in UED experiment: invoke the exact standalone map writer in Configure."""
import hashlib
import logging
from pathlib import Path
import time
import numpy as np
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain

EXPAND2_SHA256 = '1739094ffe6e80a7eae0496961730399c0f1dce077f7426e445137fac4434894'


def prepare(cfg, asics, filename):
    """Validate file and recorded metadata before any hardware changes."""
    path = Path(filename)
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPAND2_SHA256:
        raise ValueError('Standalone DAQ test requires the reviewed expand2 mask')
    mask = np.load(path, allow_pickle=False)
    if mask.shape != gain.RAW_SHAPE or not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid standalone DAQ mask')
    if asics != list(range(16)) or cfg['user']['gain_mode'] != 5:
        raise ValueError('Standalone DAQ test requires all 16 ASICs and MapFHL')
    pixels = np.asarray(cfg['user']['pixel_map_raw']).reshape(gain.RAW_SHAPE)
    np.testing.assert_array_equal(pixels, np.where(mask, 8, 12))
    if any(cfg['expert']['EpixQuad'][f'Epix10kaSaci{i}']['trbit'] != 1 for i in asics):
        raise ValueError('Standalone DAQ test requires trbit 1')
    # Same loader, conversion, and input ordering as standalone capture.
    selected, source = gain._selected_pixels_from_map(str(path))
    np.testing.assert_array_equal(gain._expected_raw_map(gain.GAIN_MODES['mapfhl'], selected), mask)
    return dict(selected=selected, asics=asics, source=source,
                mask_sha256=EXPAND2_SHA256,
                writer_path=gain.__file__,
                writer_sha256=hashlib.sha256(Path(gain.__file__).read_bytes()).hexdigest())


def program(camera, plan):
    start = time.perf_counter()
    state = gain._disable_camera_trigger(camera)
    for name, variable in (('TrigEn', camera.SystemRegs.TrigEn),
                           ('AutoTrigEn', camera.SystemRegs.AutoTrigEn),
                           ('RdoutEn', camera.RdoutCore.RdoutEn)):
        if state[name] is None or variable.get(read=True):
            raise RuntimeError('Cannot establish quiescent camera for standalone writer')
    logging.info('STANDALONE_DAQ_WRITER begin mask=%s selected=%d helper=%s sha256=%s',
                 plan['mask_sha256'], len(plan['selected']), plan['writer_path'], plan['writer_sha256'])
    # This is the same function invoked by capture_gain_writer.program().
    # It fills every ASIC with FH, writes all selected FL pixels in input order,
    # then disables ASIC IsEn/local access. No 44-pixel correction is performed.
    gain._write_gain_mode(camera, gain.GAIN_MODES['mapfhl'], plan['asics'],
                          plan['selected'], [], 0, verbose=False)
    # On failure above, gates remain off and Configure fails. On success restore
    # the DAQ's original camera gates before its normal Run-trigger enable.
    gain._restore_camera_trigger(camera, state)
    logging.info('STANDALONE_DAQ_WRITER complete elapsed=%.3f s; no targeted repair',
                 time.perf_counter()-start)
