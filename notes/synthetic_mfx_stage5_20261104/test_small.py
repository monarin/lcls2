"""Exercise the full writer/validator with an explicitly test-only 9-event schedule."""
import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import make_pilot
import verify_pilot
from xtc_fixture import load_sources, save_json, schedule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sources = load_sources()
    with patch.object(make_pilot, 'COUNTS', [2, 3, 4]), patch.object(make_pilot, 'EVENTS', 9), \
         patch.object(make_pilot, 'schedule', partial(schedule, counts=[2, 3, 4])), \
         patch.object(verify_pilot, 'COUNTS', [2, 3, 4]), patch.object(verify_pilot, 'EVENTS', 9), \
         patch.object(verify_pilot, 'schedule', partial(schedule, counts=[2, 3, 4])):
        make_pilot.make(args.root, sources)
        result = verify_pilot.verify(args.root, sources)
        result['scope'] = 'Test-only 9-event fixture; not full generation or publication acceptance'
        save_json(args.report, result)
