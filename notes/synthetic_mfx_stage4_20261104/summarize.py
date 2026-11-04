"""Produce compact review evidence without duplicating per-event reports in Git."""
import json
from common import HERE, SAMPLES, STAGE1, require, save, sha


def available(value):
    return value is not None and not (isinstance(value, dict) and "source_api_exception" in value)


def main():
    source = json.loads((HERE/'source.json').read_text())
    pilot = json.loads((HERE/'pilot.json').read_text())
    require(source['status'] == pilot['status'] == 'pass', 'Serial reports not passing')
    require(source['events'] == pilot['events'], 'Serial reports differ')
    require(source['metadata'] == pilot['metadata'] and source['masks'] == pilot['masks'], 'Metadata/masks differ')
    refs = {}
    for name in ('source.json', 'pilot.json'):
        r = json.loads((HERE/name).read_text())
        refs[name] = {'sha256': sha((HERE/name).read_bytes()), 'bytes': (HERE/name).stat().st_size,
                      'runtime': r['runtime'], 'calibration': r['calibration']}
    first = source['events'][0]
    summary = {'status': 'pass', 'reference_reports': refs, 'steps': pilot['steps'],
               'event_count': len(source['events']), 'flat_counts': pilot['flat_counts'],
               'fields_per_event': len(first['fields']), 'epics_per_event': len(first['epics']),
               'fields_with_nonmissing_values': sorted(k for k in first['fields'] if any(available(e['fields'][k]) for e in source['events'])),
               'epics_with_nonmissing_values': sum(any(e['epics'][k] is not None for e in source['events']) for k in first['epics']),
               'samples': {str(i): source['events'][i]['calibrated'] for i in SAMPLES},
               'masks': source['masks'], 'calibration_options': source['calib_options'],
               'all_event_fields_epics_damage_equal': True,
               'known_encoder_exceptions': sum(isinstance(e['fields']['MfxUsbEncoder01/raw/descriptions'], dict) and 'source_api_exception' in e['fields']['MfxUsbEncoder01/raw/descriptions'] for e in source['events']),
               'manifest_sha256': sha((STAGE1/'manifest.json').read_bytes()),
               'deployment_journal_sha256': sha((STAGE1/'stage2_deployment_journal.json').read_bytes())}
    save(HERE/'serial_summary.json', summary)


if __name__ == '__main__':
    main()
