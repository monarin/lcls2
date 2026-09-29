"""Compare saved UED history with commissioning files, entirely offline."""
import csv
import json
from pathlib import Path

import numpy as np
from pyrogue.utilities.fileio import FileReader
from psdaq.configdb.epixquad_layout import rogue_payload_to_daq_raw
from psdaq.debugtools.epixquad1kfps.capture_standalone import check_file_integrity
from psdaq.debugtools.epixquad1kfps.standalone_profile import file_sha256

folder = Path(__file__).resolve().parent
history_file = folder/'history-fixed-gain-evidence.json'
history = json.loads(history_file.read_text())
pattern = history['fl_coordinate_pattern']
historical_pixels = {(pattern['segment'], row, col)
                     for row in range(pattern['rows_inclusive'][0], pattern['rows_inclusive'][1]+1)
                     for col in pattern['columns']}
with (folder/'fl-persistent-mismatch-candidates.csv').open() as stream:
    present_pixels = {tuple(int(r[k]) for k in ('segment', 'raw_row', 'raw_col'))
                      for r in csv.DictReader(stream)}
assert len(historical_pixels) == 352
assert present_pixels == historical_pixels
current = []
for name in ('attempt02-fh', 'attempt03-fm', 'attempt04-fl'):
    attempt = folder/name
    record = json.loads((attempt/'test.json').read_text())
    data_file = attempt/'capture.dat'
    check_file_integrity(data_file)
    checksum = file_sha256(data_file)
    assert checksum == record['data']['sha256']
    histograms = []
    adc_means = []
    for header, data in FileReader(str(data_file)).records():
        payload = np.asarray(data).view(np.uint8).tobytes()
        if header.channel != 1 or header.error or len(payload) not in (1095248, 1095288):
            continue
        raw, _ = rogue_payload_to_daq_raw(payload)
        histograms.append(np.bincount(((raw >> 14) & 3).ravel(), minlength=4))
        adc_means.append(float((raw & 0x3fff).mean()))
    assert len(histograms) == record['analysis']['valid_images'] == 50
    current.append({'attempt': name, 'mode': record['mode'],
                    'valid_images': len(histograms), 'data_sha256': checksum,
                    'top2_counts_all': np.sum(histograms, axis=0).tolist(),
                    'top2_counts_first': histograms[0].tolist(),
                    'top2_counts_frames_2_to_50': np.sum(histograms[1:], axis=0).tolist(),
                    'raw14_mean_all': float(np.mean(adc_means)),
                    'raw14_mean_first': adc_means[0],
                    'raw14_mean_frames_2_to_50': float(np.mean(adc_means[1:])),
                    'analysis': record['analysis']})

# Also compare the earlier mixed-map FN coordinates from experiment run 2.
map_csv = Path('/sdf/home/m/monarin/tmp/epix-per-pixel-gainmode/per_pixel_gain_diag/20260706/run2_fn_raw14_delta_to_fh_fm_fl_pedestals.csv')
with map_csv.open() as stream:
    map_pixels = {tuple(int(r[k]) for k in ('segment', 'raw_row', 'raw_col'))
                  for r in csv.DictReader(stream)}
result = {'history_evidence_sha256': file_sha256(history_file),
          'current_captures': current,
          'persistent_fl_match': {'historical_pixels': len(historical_pixels),
                                 'current_pixels': len(present_pixels),
                                 'intersection': len(historical_pixels & present_pixels),
                                 'new_persistent_pixels': len(present_pixels-historical_pixels),
                                 'missing_historical_pixels': len(historical_pixels-present_pixels)},
          'historical_map_run2': {'source_csv': str(map_csv), 'sha256': file_sha256(map_csv),
                                 'fn_pixels': len(map_pixels),
                                 'overlap_current_persistent': len(map_pixels & present_pixels)},
          'scope': 'Historical run summaries recovered from task history; current raw files decoded anew. No live hardware access.'}
(folder/'fixed-gain-history-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
