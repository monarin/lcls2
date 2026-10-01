"""Compare matched incoming DMA, pre-write datagrams and finalized XTC2 events."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from psana import dgram
from psdaq.configdb.epixquad_layout import (
    RAW_SHAPE, ROGUE_IMAGE_PAYLOAD_BYTES, event_batcher_subframes, rogue_payload_to_daq_raw,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def header(data):
    if len(data) < 24:
        raise ValueError('Short XTC2 datagram')
    size = 12 + struct.unpack_from('<I', data, 20)[0]
    if not 24 <= size <= 64*1024**2:
        raise ValueError('Invalid XTC2 extent')
    timestamp, env = struct.unpack_from('<QI', data)
    return size, (env >> 24) & 15, timestamp


def image(data, config):
    size, service, _ = header(data)
    if size != len(data) or service != 12:
        raise ValueError('Expected exactly one L1Accept datagram')
    dg = dgram.Dgram(view=data, config=config)
    segments = getattr(dg, 'epixquad1kfps', {})
    if set(segments) != set(range(4)):
        raise ValueError('Missing/unexpected detector segments')
    raw = np.stack([segments[s].raw.raw for s in range(4)])
    if raw.shape != RAW_SHAPE:
        raise ValueError('Incomplete raw image')
    damage = [int(dg._xtc.damage)]
    for seg in segments.values():
        damage.extend((int(seg._xtc.damage), int(seg.raw._xtc.damage)))
    return raw, damage


def read_disk(paths, targets, config_bytes):
    matched, files = {}, []
    endrun = False
    for path in paths:
        if str(path).endswith('.inprogress'):
            raise ValueError('End the run and use finalized XTC2 files')
        initial = path.stat()
        digest = hashlib.sha256()
        configured = False
        offset = 0
        with path.open('rb') as stream:
            while prefix := stream.read(24):
                size, service, ts = header(prefix)
                payload = prefix + stream.read(size-24)
                if len(payload) != size:
                    raise ValueError('Truncated disk datagram')
                digest.update(payload)
                if service == 2:
                    if payload != config_bytes:
                        raise ValueError('Disk Configure differs from diagnostic Configure')
                    configured = True
                if service == 12 and ts in targets:
                    if not configured or ts in matched:
                        raise ValueError('Missing Configure or duplicate matching timestamp')
                    matched[ts] = (payload, str(path), offset)
                endrun |= service == 5
                offset += size
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise ValueError('Disk file changed during comparison')
        files.append(dict(path=str(path), bytes=offset, sha256=digest.hexdigest()))
    return matched, files, endrun


def compare(folder, paths, output):
    evidence = Path(__file__).resolve().parents[2] / 'ued-commissioning-20260928'
    mask_path = evidence/'map-inputs/expand2.npy'
    known_path = evidence/'fl-persistent-mismatch-candidates.csv'
    expected = np.load(mask_path, allow_pickle=False).astype(bool)
    known = np.zeros(RAW_SHAPE, bool)
    with known_path.open() as stream:
        for row in csv.DictReader(stream):
            known[tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))] = True
    meta_bytes = (folder/'manifest.json').read_bytes()
    meta = json.loads(meta_bytes)
    if meta['schema_version'] != 1 or meta['status'] != 'complete':
        raise ValueError('Diagnostic capture incomplete; inspect manifest before comparing')
    samples = sorted(meta['samples'], key=lambda s: s['timestamp'])
    if len(samples) != meta['requested_frames'] or len(samples) < 2:
        raise ValueError('Need the complete requested sample and at least two events')
    targets = {s['timestamp'] for s in samples}
    if len(targets) != len(samples):
        raise ValueError('Duplicate diagnostic timestamps')
    config_bytes = (folder/'configure.xtc2').read_bytes()
    if header(config_bytes)[:2] != (len(config_bytes), 2):
        raise ValueError('Invalid diagnostic Configure')
    config = dgram.Dgram(view=config_bytes)
    pixels = np.stack([config.epixquad1kfps[s].config.asicPixelConfig for s in range(4)])
    trbits = np.stack([config.epixquad1kfps[s].config.trbit for s in range(4)])
    np.testing.assert_array_equal(pixels, np.where(expected, 8, 12))
    np.testing.assert_array_equal(trbits, np.ones((4,4), dtype=np.uint8))
    identity = json.loads((evidence/'test-attempt04.json').read_text())['camera_identity']['carrier_ids']
    detids = [config.software.epixquad1kfps[s].detid for s in range(4)]
    if [[int(v) for v in ident.split('-')[1:3]] for ident in detids] != identity:
        raise ValueError('Known-bad mask belongs to a different detector')
    disk, files, endrun = read_disk(paths, targets, config_bytes)
    counts = {f'{stage}_{kind}': np.zeros(RAW_SHAPE, np.uint32)
              for stage in ('input','prewrite','disk') for kind in ('FP','FN')}
    differences = {key: np.zeros(RAW_SHAPE, np.uint32)
                   for key in ('input_vs_prewrite','prewrite_vs_disk')}
    rows, assessed = [], 0
    for index, sample in enumerate(samples):
        row = dict(sample)
        row.update(first_sample=index == 0, assessed=False, errors=[])
        try:
            incoming = (folder/sample['input']).read_bytes()
            before = (folder/sample['prewrite']).read_bytes()
            if len(incoming) != sample['input_bytes'] or len(before) != sample['prewrite_bytes']:
                raise ValueError('Diagnostic file length does not match manifest')
            if not sample['submitted_to_file_writer']:
                raise ValueError('Event was not selected for disk writing')
            if sample['timestamp'] not in disk:
                raise ValueError('No matching disk event')
            recorded, filename, offset = disk[sample['timestamp']]
            frames = event_batcher_subframes(incoming)
            if (len(incoming) not in ROGUE_IMAGE_PAYLOAD_BYTES or 2 not in frames or
                    frames[2]['size'] != 1095232 or any(f['last_user'] & 1 for f in frames.values())):
                raise ValueError('Unsupported, incomplete or damaged incoming camera payload')
            pulse, ts = struct.unpack_from('<QQ', incoming, frames[0]['offset'])
            if pulse >> 56 != 12 or (pulse & 0x00ffffffffffffff) != sample['pulse_id'] or ts != sample['timestamp']:
                raise ValueError('Incoming timing does not match DRP identity')
            if header(before)[2] != ts:
                raise ValueError('Pre-write timestamp does not match incoming data')
            raw_input, skip = rogue_payload_to_daq_raw(incoming)
            if skip*2 != frames[2]['offset']+32:
                raise ValueError('Decoder offset disagrees with camera subframe location')
            raw_before, before_damage = image(before, config)
            raw_disk, disk_damage = image(recorded, config)
            row.update(disk_file=filename, disk_offset=offset,
                       input_sha256=sha(incoming), prewrite_sha256=sha(before), disk_sha256=sha(recorded),
                       prewrite_disk_bytes_equal=before == recorded,
                       prewrite_damage=before_damage, disk_damage=disk_damage)
            if any(before_damage) or any(disk_damage):
                raise ValueError('Damaged event; excluded from gain assessment')
            raw = dict(input=raw_input, prewrite=raw_before, disk=raw_disk)
            row['gainbits'] = {}
            for stage, array in raw.items():
                low = (array & 0x4000) != 0
                fp, fn = low & ~expected, ~low & expected
                row['gainbits'][stage] = dict(FP=int(fp.sum()), FN=int(fn.sum()),
                    FP_outside_known=int(fp[~known].sum()), FN_outside_known=int(fn[~known].sum()))
                if index:
                    counts[f'{stage}_FP'] += fp
                    counts[f'{stage}_FN'] += fn
            for key, a, b in (('input_vs_prewrite',raw_input,raw_before),
                              ('prewrite_vs_disk',raw_before,raw_disk)):
                different = a != b
                row[key+'_different_pixel_words'] = int(different.sum())
                if index:
                    differences[key] += different
            row['assessed'] = index > 0
            assessed += int(index > 0)
        except (ValueError, KeyError, AttributeError, IndexError) as exc:
            row['errors'].append(str(exc))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'after-first-counts.npz', **counts, **differences, known_bad=known)
    with (output/'gain-mismatches.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['stage','kind','segment','raw_row','raw_col','occurrences','assessed_frames','within_known_bad'])
        for key, array in counts.items():
            stage, kind = key.split('_')
            for coord in np.argwhere(array):
                coord = tuple(int(x) for x in coord)
                writer.writerow([stage,kind,*coord,int(array[coord]),assessed,int(known[coord])])
    result = dict(diagnostic=str(folder), manifest_sha256=sha(meta_bytes), configure_sha256=sha(config_bytes),
        mask_sha256=sha(mask_path.read_bytes()), known_bad_sha256=sha(known_path.read_bytes()),
        detector_ids=detids, disk_files=files, endrun_seen=endrun, sampled=len(samples), assessed=assessed,
        policy='Earliest sampled image retained but excluded from gain assessment; all pairs compared bytewise',
        identity='Incoming pulse ID and timestamp checked against hook; XTC2 matched by unique timestamp (no pulse ID in Dgram)',
        complete_comparison=endrun and all(not row['errors'] for row in rows),
        all_prewrite_disk_bytes_equal=all(row.get('prewrite_disk_bytes_equal',False) for row in rows),
        after_first={key: dict(unique_pixels=int(np.count_nonzero(array)),
                              outside_known_unique=int(np.count_nonzero(array[~known])),
                              occurrences=int(array.sum())) for key,array in counts.items()},
        word_differences={key: int(array.sum()) for key,array in differences.items()}, frames=rows)
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('xtc', type=Path, nargs='+', help='Finalized detector stream chunks, including EndRun')
    args = parser.parse_args()
    result = compare(args.diagnostic, args.xtc, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}, indent=2))
    raise SystemExit(0 if result['complete_comparison'] else 2)
