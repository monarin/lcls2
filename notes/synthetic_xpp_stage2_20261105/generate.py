"""Generate only run-3 pilot or eight-event seam fixture, in private staging."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

from fixture import HERE, ROOT, RUN, create_root, load_sources, plan, private_dir, private_output, rewrite, wire, check_stat


def generate(mode):
    create_root()
    sources = load_sources()
    mapping = plan(sources[0]['index'],mode)
    folder = ROOT / mode
    private_dir(folder)
    private_dir(folder/'smalldata')

    def stream(source):
        n = source['stream']
        names = dict(bd=f'prjdatpsana-r{RUN:04d}-s{n:03d}-c000.xtc2',
                     smd=f'smalldata/prjdatpsana-r{RUN:04d}-s{n:03d}-c000.smd.xtc2')
        hashes = {k:hashlib.sha256() for k in names}
        with Path(source['entry']['bigdata']['path']).open('rb') as original, \
                private_output(folder/names['bd']) as bd, private_output(folder/names['smd']) as smd:
            for source_record,timestamp in zip(mapping['source_record'],mapping['timestamp']):
                row = source['index'][source_record]
                raw = wire.read_record(original,int(row['bd_offset']))
                p,nbytes = int(row['smd_offset']),int(row['smd_size'])
                raw_smd = source['smd'][p:p+nbytes]
                wire.require(len(raw)==row['bd_size'] and wire.header(raw)[:2]==wire.header(raw_smd)[:2], 'Source record differs')
                if int(row['service'])==10:
                    wire.require(raw==raw_smd, 'SlowUpdate bigdata/SMD mismatch')
                new = rewrite(raw,timestamp,source['bd_schemas'])
                new_smd = rewrite(raw_smd,timestamp,source['smd_schemas'],offset=bd.tell(),size=len(new))
                bd.write(new); smd.write(new_smd)
                hashes['bd'].update(new); hashes['smd'].update(new_smd)
            for output in (bd,smd):
                output.flush(); os.fsync(output.fileno())
        check_stat(source['entry']['bigdata'])
        files = {k:dict(relative_path=name,bytes=(folder/name).stat().st_size,sha256=hashes[k].hexdigest()) for k,name in names.items()}
        wire.require(all((folder/name).stat().st_mode & 0o077 == 0 for name in names.values()), 'Output file is not private')
        print('Generated',mode,'stream',n,'records',len(mapping['source_record']),flush=True)
        return dict(stream=n,files=files)

    with ThreadPoolExecutor(max_workers=4) as pool:
        streams = list(pool.map(stream,sources))
    serial_mapping = {k:v.tolist() if hasattr(v,'tolist') else v for k,v in mapping.items()}
    generation = dict(status='generated',created_utc=datetime.now(timezone.utc).isoformat(),
                      run=RUN,folder=str(folder),mapping=serial_mapping,streams=streams,
                      total_bytes=sum(f['bytes'] for s in streams for f in s['files'].values()),
                      code_sha256={name:wire.file_sha(HERE/name) for name in ('fixture.py','generate.py')})
    wire.save_json(folder/'generation.json',generation)
    summary = {k:v for k,v in generation.items() if k!='mapping'}
    summary.update(events=mapping['events'],records_per_stream=len(mapping['timestamp']),
                   generation_sha256=wire.file_sha(folder/'generation.json'))
    wire.save_json(HERE/f'{mode}_generation.json',summary)
    print('Completed',mode,generation['total_bytes'],'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('pilot','seam'))
    generate(parser.parse_args().mode)
