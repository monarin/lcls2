"""Read-only final-link, unchanged-run-1 and calibrated first-event check."""
import argparse
import json
import os
from pathlib import Path
import requests

from acceptance import fingerprint
from xtc_fixture import DEST, DEST_EXP, RUN, check_stat, file_sha, require, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0' and os.environ.get('PS_PARALLEL') == 'none', 'Wrong flags')
    original_request = requests.sessions.Session.request

    def read_only(self, method, url, *a, **kw):
        require(method.upper() == 'GET', 'Final check forbids HTTP mutations')
        kw.setdefault('timeout', 180)
        return original_request(self, method, url, *a, **kw)

    requests.sessions.Session.request = read_only
    here = Path(__file__).resolve().parent
    before = json.loads((here/'preflight.json').read_text())
    require(len(before['run1_files']) == 20, 'Incomplete run-1 baseline')
    for path, stat in before['run1_files'].items():
        check_stat(Path(path), stat)
    gen = json.loads((args.staging/'generation.json').read_text())
    linked = []
    for stream in gen['streams']:
        for item in stream['files'].values():
            relative = item['relative_path']
            original, final = args.staging/relative, DEST/relative
            a, b = original.stat(), final.stat()
            require(not final.is_symlink() and (a.st_dev,a.st_ino,a.st_size) == (b.st_dev,b.st_ino,item['size']), 'Wrong final link')
            linked.append(str(final))
    require(len(linked) == 20 and len(set(linked)) == 20, 'Incomplete final files')
    require(set(linked) == {str(p) for folder in (DEST, DEST/'smalldata') for p in folder.glob('*r0002*.xtc2*')}, 'Unexpected final run-2 files')
    source = json.loads(args.reference.read_text())
    require(source['status'] == 'pass' and source['kind'] == 'source', 'Wrong source reference')
    first = min(source['samples'], key=int)
    from psana import DataSource
    runs = events = 0
    for run in DataSource(exp=DEST_EXP,run=RUN,dir=str(DEST),live=False,max_events=1).runs():
        runs += 1
        require(run.expt == DEST_EXP and run.runnum == RUN, 'Wrong run')
        jf, step = run.Detector('jungfrau'), run.Detector('step_value')
        for evt in run.events():
            events += 1
            require(str(int(evt.timestamp)) == first and step(evt) == 0, 'Wrong first event/step')
            sample = {'raw':fingerprint(jf.raw.raw(evt)), 'calib':fingerprint(jf.raw.calib(evt,cversion=3,cmpars=None))}
            require(sample == source['samples'][first], 'Published calibrated sample differs')
    require(runs == events == 1, 'Wrong smoke counts')
    save_json(args.report, {'status':'pass','run1_files_unchanged':20,'published_links':linked,
                           'first_timestamp':first,'sample':sample,'source_sha256':file_sha(args.reference),
                           'script_sha256':file_sha(Path(__file__))})
    print('PASS published smoke, 20 final links, 20 unchanged run-1 files',flush=True)


if __name__ == '__main__':
    main()
