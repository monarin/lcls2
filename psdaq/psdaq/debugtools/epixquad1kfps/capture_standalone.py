#!/usr/bin/env python3
"""Profiled headless ePix capture with a new directory for every attempt."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import shutil
import signal
import socket
import struct
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hardware
from psdaq.debugtools.epixquad1kfps import capture_gain_writer as gain_writer
from psdaq.debugtools.epixquad1kfps.standalone_profile import (
    file_sha256, software_snapshot, validate_hardware,
)
from psdaq.configdb.epixquad_layout import rogue_payload_to_daq_raw
from psdaq.debugtools.epixquad1kfps.export_mismatches import export_mismatches


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


@contextmanager
def output_log(path):
    """Capture Python and native Rogue stdout/stderr, restoring both on exit."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with path.open('w') as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


class ImageCounts:
    """Count successfully forwarded full images separately from other records."""
    def __init__(self, sizes, max_bytes):
        self.sizes = set(sizes)
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.valid = self.records = self.invalid = self.bytes = 0
        self.fault = None
        self.last_receive = time.monotonic()
        self.first_image_receive = self.last_image_receive = None

    def room(self, size):
        with self.lock:
            if self.bytes + size + 8 > self.max_bytes:
                self.fault = 'Capture byte limit reached'
                return False
            return True

    def observed(self, payload, error):
        valid = False
        if not error and len(payload) in self.sizes:
            try:
                rogue_payload_to_daq_raw(payload)
                valid = True
            except (ValueError, IndexError):
                pass
        with self.lock:
            self.records += 1
            self.valid += int(valid)
            self.invalid += int(not valid)
            self.bytes += len(payload) + 8
            self.last_receive = time.monotonic()
            if valid:
                if self.first_image_receive is None:
                    self.first_image_receive = self.last_receive
                self.last_image_receive = self.last_receive

    def snapshot(self):
        with self.lock:
            span = (self.last_image_receive - self.first_image_receive) if self.valid > 1 else 0
            return {'valid_images': self.valid, 'records': self.records,
                    'invalid_records': self.invalid, 'bytes': self.bytes,
                    'image_arrival_rate_hz': (self.valid - 1) / span if span > 0 else None,
                    'fault': self.fault, 'last_receive_monotonic': self.last_receive}


def make_stream_writer(camera, path, counts):
    import rogue.interfaces.stream as ris
    import rogue.utilities.fileio

    class Sink(ris.Slave, ris.Master):
        def __init__(self):
            ris.Slave.__init__(self)
            ris.Master.__init__(self)
            self.armed = False
            self.callback_lock = threading.Lock()

        def disarm(self):
            # Wait for any in-flight forwarding before closing the native file.
            with self.callback_lock:
                self.armed = False

        def _acceptFrame(self, frame):
            with self.callback_lock:
                self.accept(frame)

        def accept(self, frame):
            if not self.armed:
                return
            try:
                with frame.lock():
                    payload = bytearray(frame.getPayload())
                    frame.read(payload, 0)
                    error = frame.getError()
                if counts.room(len(payload)):
                    self._sendFrame(frame)
                    counts.observed(payload, error)
            except Exception as exc:
                with counts.lock:
                    counts.fault = 'Stream callback failed: '+repr(exc)

    writer = rogue.utilities.fileio.StreamWriter()
    writer.setDropErrors(False)
    sink = Sink()
    sink >> writer.getChannel(1)
    try:
        writer.open(str(path))
        camera.pgpVc0 >> sink
        sink.armed = True
    except BaseException:
        sink.disarm()
        writer.close()
        raise
    return writer, sink


def quiesce(camera):
    """Attempt every disable even when an earlier register access fails."""
    errors = []
    for variable in (camera.SystemRegs.TrigEn, camera.SystemRegs.AutoTrigEn,
                     camera.RdoutCore.RdoutEn):
        try:
            variable.set(False)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError('Could not disable camera: '+ '; '.join(errors))


def wait_for_images(counts, frames, timeout, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout
    while True:
        state = counts.snapshot()
        if state['fault']:
            raise RuntimeError(state['fault'])
        if state['valid_images'] >= frames:
            return
        if clock() >= deadline:
            raise TimeoutError('Timed out with %d/%d valid images' % (state['valid_images'], frames))
        sleep(0.02)


def trigger_snapshot(camera):
    """Actual camera settings; rate requests do not change external timing."""
    regs = camera.SystemRegs
    return {'source': int(regs.TrigSrcSel.get(read=True)),
            'auto_period_ticks': int(regs.AutoTrigPer.get(read=True)),
            'auto_frequency_hz': float(regs.AutoTrigFreqHz.get(read=True))}


def capture(args, profile, selected, outdir, record):
    description = record.setdefault('gain_writer', gain_writer.describe_writer(args.gain_writer))
    map_writer = gain_writer.resolve_writer(description)
    rates = args.rate_sequence_hz or [profile['rate_hz']]

    def identity(value):
        record['pcie_identity'] = value
        record['runtime'] = hardware.runtime_snapshot()
        write_json(outdir/'test.json', record)

    with hardware.session(profile, camera_vc_mask=0x3, on_identity=identity) as (pcie, camera):
        for index, rate in enumerate(rates):
            phase_dir, phase = outdir, record
            if args.rate_sequence_hz:
                phase_dir = outdir / ('phase%02d' % (index + 1))
                phase_dir.mkdir(exist_ok=False)
                phase = {k: v for k, v in record.items() if k not in ('phases', 'inputs')}
                phase.update(test_id=phase_dir.name, status='running',
                             parent_test_id=record['test_id'], phase_index=index + 1,
                             requested_rate_hz=rate, map_written_this_phase=False,
                             map_write_phase='phase01', path=phase_dir.name,
                             provenance_directory='..')
                record.setdefault('phases', []).append(phase)
                shutil.copyfile(outdir/'expected-map.npy', phase_dir/'expected-map.npy')
            try:
                acquire_phase(args, profile, selected, phase_dir, phase, pcie, camera,
                              map_writer, rate, program_map=index == 0)
                phase['status'] = 'completed'
            except BaseException as exc:
                phase['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                phase['error'] = repr(exc)
                raise
            finally:
                phase['finished_utc'] = utc_now()
                if phase is not record:
                    write_json(phase_dir/'test.json', phase)
                write_json(outdir/'test.json', record)


def acquire_phase(args, profile, selected, outdir, record, pcie, camera,
                  map_writer, rate_hz, program_map):
    """Bounded capture; later phases reuse the configured map in this session."""
    counts = ImageCounts(profile['image_payload_bytes'], args.max_bytes)
    record['map_written_this_phase'] = False
    writer = sink = None
    try:
        record['runtime'] = hardware.runtime_snapshot()
        quiesce(camera)
        if program_map:
            hardware.load_camera_baseline(camera, profile['camera_yaml'], args.asics)
            quiesce(camera)  # YAML can re-enable triggering.
        record['camera_identity'] = hardware.camera_identity(camera)
        write_json(outdir/'test.json', record)
        if not record['camera_identity']['adc_done'] or record['camera_identity']['adc_failed']:
            raise RuntimeError('Camera ADC status is not done/pass')
        if program_map:
            record['map_write_started_utc'] = utc_now()
            write_json(outdir/'test.json', record)
            gain_writer.program(camera, args, selected, record['gain_writer'], map_writer)
            record['map_write_finished_utc'] = utc_now()
            record['map_written_this_phase'] = True
        record['requested_rate_hz'] = rate_hz
        camera.SystemRegs.TrigSrcSel.set(profile['camera_trigger_source'])
        if profile['camera_auto_trigger']:
            camera.SystemRegs.AutoTrigFreqHz.set(rate_hz)
        record['trigger_readback'] = trigger_snapshot(camera)
        write_json(outdir/'test.json', record)
        writer, sink = make_stream_writer(camera, outdir/'capture.dat', counts)
        record['acquisition_started_utc'] = utc_now()
        pcie.StartRun()
        camera.RdoutCore.RdoutEn.set(True)
        camera.SystemRegs.AutoTrigEn.set(profile['camera_auto_trigger'])
        camera.SystemRegs.TrigEn.set(True)
        wait_for_images(counts, args.frames, args.timeout)
    finally:
        primary_error = sys.exc_info()[1]
        errors = []
        try:
            quiesce(camera)
        except Exception as exc:
            errors.append(repr(exc))
        try:
            # Camera-off alone leaves timing-only records flowing on VC0.
            # Disable upstream event production before waiting for quiet.
            hardware.pause_triggers(pcie)
        except Exception as exc:
            errors.append(repr(exc))
        try:
            # Let pending camera/PCIe frames reach the writer before blowoff.
            deadline = time.monotonic() + args.drain_timeout
            while counts.snapshot()['records'] and time.monotonic() - counts.snapshot()['last_receive_monotonic'] < args.drain_seconds:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Capture stream did not drain')
                time.sleep(0.02)
        except Exception as exc:
            errors.append(repr(exc))
        try:
            pcie.StopRun()
        except Exception as exc:
            errors.append(repr(exc))
        if sink is not None:
            try:
                sink.disarm()
            except Exception as exc:
                errors.append(repr(exc))
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                errors.append(repr(exc))
        record['stream_counts'] = counts.snapshot()
        record['cleanup_errors'] = errors
        record['acquisition_stopped_utc'] = utc_now()
        if errors and primary_error is None:
            raise RuntimeError('Capture cleanup failed: '+ '; '.join(errors))


def check_file_integrity(path):
    """FileReader can stop at an incomplete trailing record; reject truncation."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        while stream.tell() < size:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError('Truncated Rogue file header')
            length, _, _, _ = struct.unpack('<IHBB', header)
            if length < 4 or stream.tell() + length - 4 > size:
                raise ValueError('Invalid or truncated Rogue record')
            stream.seek(length - 4, 1)


def analyze(path, expected, sizes, outdir):
    check_file_integrity(path)
    from pyrogue.utilities.fileio import FileReader
    from psdaq.debugtools.epixquad1kfps.read_xpmmini_rogue_file import _raw_area_masks
    fp = np.zeros(expected.shape, dtype=np.uint32)
    fn = np.zeros(expected.shape, dtype=np.uint32)
    valid = invalid = 0
    for header, data in FileReader(str(path)).records():
        if header.channel != 1:
            continue
        payload = np.asarray(data).view(np.uint8).tobytes()
        if header.error or len(payload) not in sizes:
            invalid += 1
            continue
        raw, _ = rogue_payload_to_daq_raw(payload)
        low = (raw & 0x4000) != 0
        fp += low & ~expected
        fn += ~low & expected
        if not valid:
            np.save(outdir/'first_raw.npy', raw)
        valid += 1
    np.savez_compressed(outdir/'mismatches.npz', fp=fp, fn=fn)
    usable, control = _raw_area_masks()
    return {'valid_images': valid, 'invalid_records': invalid,
            'false_positive_pixels': int(np.count_nonzero(fp[usable])),
            'false_negative_pixels': int(np.count_nonzero(fn[usable])),
            'false_positive_occurrences': int(fp[usable].sum()),
            'false_negative_occurrences': int(fn[usable].sum()),
            'control_row_fp_occurrences': int(fp[control].sum()),
            'control_row_fn_occurrences': int(fn[control].sum())}


def finalize_capture(outdir, record, expected, profile, args):
    """Analyze even partial files, preserving a separate record for each phase."""
    exit_code = 0
    data = outdir/'capture.dat'
    if record['status'] == 'completed' and not data.is_file():
        record['status'] = 'failed'
        record['error'] = 'Capture returned without a data file'
        exit_code = 1
    if data.exists():
        try:
            record['data'] = {'path': data.name, 'size': data.stat().st_size,
                              'sha256': file_sha256(data)}
            record['analysis'] = analyze(data, expected, profile['image_payload_bytes'], outdir)
            if record['status'] == 'completed' and record['analysis']['valid_images'] < args.frames:
                raise RuntimeError('Recorded file has fewer valid images than requested')
        except Exception as exc:
            record['analysis_error'] = repr(exc)
            record['status'] = 'failed'
            traceback.print_exc()
            exit_code = 1
    record['finished_utc'] = utc_now()
    if 'analysis' in record:
        try:
            record['mismatch_artifacts'] = export_mismatches(
                outdir, outdir/'mismatch-records', record)
        except Exception as exc:
            record['mismatch_export_error'] = repr(exc)
            record['status'] = 'failed'
            traceback.print_exc()
            exit_code = 1
    write_json(outdir/'test.json', record)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True, type=Path, help='New directory for this attempt; must not exist')
    parser.add_argument('--label', required=True, help='Test step or condition')
    parser.add_argument('--repetition', type=int, default=1)
    parser.add_argument('--notes', default='')
    parser.add_argument('--frames', type=int, default=50, help='Full images requested per capture/rate phase')
    parser.add_argument('--timeout', type=float, default=30, help='Image acquisition timeout per phase, seconds')
    parser.add_argument('--drain-seconds', type=float, default=0.25)
    parser.add_argument('--drain-timeout', type=float, default=5)
    parser.add_argument('--max-bytes', type=int, default=512 * 1024 * 1024,
                        help='Native file byte limit per capture/rate phase')
    parser.add_argument('--gain-writer', choices=('standalone', 'daq'), default='standalone',
                        help='DAQ selects the actual configdb raw-map helper, not full Configure')
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument('--rate-hz', type=float, default=None)
    rates.add_argument('--rate-sequence-hz', type=float, nargs='+',
                       help='Internal camera rates in one session with one map write; frames per phase')
    args = gain._parse_args(argv, parser)
    if not args.profile:
        parser.error('--profile is required')
    if args.frames <= 0 or args.max_bytes <= 0 or args.repetition <= 0:
        parser.error('frames, max-bytes and repetition must be positive')
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.drain_seconds, args.drain_timeout)):
        parser.error('timeouts must be positive and finite')
    if args.drain_seconds >= args.drain_timeout:
        parser.error('--drain-seconds must be less than --drain-timeout')
    if args.inject_pixel or args.asics != list(range(16)) or args.readback_pixel_mask_asic is not None:
        parser.error('Capture currently supports all 16 ASICs without charge injection or bitmap readback')
    if args.save_expected_map or args.leave_trigger_off or args.load_ued_yaml:
        parser.error('Capture owns expected-map output, trigger cleanup and profile camera-YAML loading')
    if args.camera_vc_mask != 0x2:
        parser.error('Capture opens VC0 and VC1 itself; omit --camera-vc-mask')
    selected, selected_source = gain._selected_pixels(args)
    expected = gain._expected_raw_map(args.mode, selected).astype(bool)
    if args.gain_writer == 'daq':
        gain_writer.pixel_config_map(args.mode, selected)  # Reject unrepresentable pixels before hardware access.
    profile = args.profile_data
    requested_rates = args.rate_sequence_hz or ([args.rate_hz] if args.rate_hz is not None else [])
    if any(not math.isfinite(rate) or rate <= 0 for rate in requested_rates):
        parser.error('Rates must be positive and finite')
    if any(not 1 <= 156250000.0 / rate <= 0xffffffff for rate in requested_rates):
        parser.error('Rate cannot be represented by the 32-bit camera auto-trigger period')
    if requested_rates and (not profile['camera_auto_trigger'] or profile['camera_trigger_source'] != 3):
        parser.error('Rate overrides require internal camera auto triggering (source 3)')
    if args.rate_sequence_hz and len(args.rate_sequence_hz) < 2:
        parser.error('--rate-sequence-hz requires at least two phases')
    if args.rate_hz is not None:
        profile['rate_hz'] = args.rate_hz
        profile['overrides']['rate_hz'] = args.rate_hz
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'test_id': outdir.name, 'status': 'preparing',
              'started_utc': utc_now(), 'host': socket.gethostname(),
              'label': args.label, 'repetition': args.repetition, 'notes': args.notes,
              'argv': sys.argv[1:] if argv is None else list(argv),
              'mode': vars(args.mode), 'asics': args.asics, 'selected_pixels': selected,
              'selected_map_source': selected_source, 'requested_frames': args.frames,
              'timeout_seconds': args.timeout, 'max_bytes': args.max_bytes,
              'write_path': 'matrix-fill plus selected-pixel writes',
              'gain_writer': gain_writer.describe_writer(args.gain_writer),
              'rate_sequence_hz': args.rate_sequence_hz,
              'requested_rate_hz': profile['rate_hz'] if not args.rate_sequence_hz else None,
              'map_write_count_planned': 1,
              'camera_vc_mask': 0x3,
              'os_release': Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,
              'uname': list(platform.uname()),
              'dry_run': args.dry_run, 'python': sys.version, 'executable': sys.executable}
    write_json(outdir/'test.json', record)
    print('Test record: '+str(outdir), flush=True)
    exit_code = 0
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt('Received signal %d' % signum)

    signal.signal(signal.SIGTERM, interrupt)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        with output_log(outdir/'console.log'):
            try:
                shutil.copyfile(args.profile, outdir/'profile.json')
                write_json(outdir/'source-profile.json', profile)
                np.save(outdir/'expected-map.npy', expected)
                inputs = {}
                for key, source in [('camera_yaml', profile['camera_yaml']),
                                    ('pcie_yaml', profile['pcie_yaml']),
                                    ('selected_map', args.selected_map)]:
                    if source:
                        source = Path(source)
                        target = outdir / (key + source.suffix)
                        shutil.copyfile(source, target)
                        inputs[key] = {'source': str(source.resolve()), 'copy': target.name,
                                       'sha256': file_sha256(target)}
                        if key in ('camera_yaml', 'pcie_yaml'):
                            profile[key] = str(target)
                record['inputs'] = inputs
                write_json(outdir/'resolved-profile.json', profile)
                bundle = os.environ.get('SUBMODULEDIR')
                if not bundle:
                    raise ValueError('Set SUBMODULEDIR before recording a test')
                snapshot = software_snapshot(Path(__file__).resolve().parents[4], bundle)
                write_json(outdir/'software.json', snapshot)
                if args.dry_run:
                    record['status'] = 'dry_run'
                else:
                    validate_hardware(profile, capture=True)
                    record['status'] = 'running'
                    write_json(outdir/'test.json', record)
                    capture(args, profile, selected, outdir, record)
                    record['status'] = 'completed'
            except BaseException as exc:
                record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                record['error'] = repr(exc)
                traceback.print_exc()
                exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            finally:
                if args.rate_sequence_hz:
                    for phase in record.get('phases', []):
                        code = finalize_capture(outdir/phase['path'], phase, expected, profile, args)
                        if code:
                            record['status'] = 'failed'
                            record['analysis_error'] = 'One or more rate phases failed analysis; see phase records'
                            if exit_code == 0:
                                exit_code = code
                    record['finished_utc'] = utc_now()
                    write_json(outdir/'test.json', record)
                else:
                    code = finalize_capture(outdir, record, expected, profile, args)
                    if exit_code == 0:
                        exit_code = code
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print('%s: %s' % (record['status'], outdir))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
