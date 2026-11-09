"""Read-only Nsight attribution; CPU ownership and GPU occupancy are separate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    end = None
    total = 0
    for a, b in sorted(intervals):
        if b <= a:
            raise ValueError('Invalid interval')
        total += b - max(a, end if end is not None else a) if end is None or b > end else 0
        end = max(b, end) if end is not None else b
    return total


def partition(ranges, begin, end):
    """Disjoint host wall ownership by innermost recognized NVTX span.

    Ranges are (start, end, label) on one thread. Non-nested overlapping spans
    are invalid. Uninstrumented gaps are explicit. This is not CPU busy time.
    """
    edges = defaultdict(list)
    edges[begin]
    edges[end]
    for i, (a, b, label) in enumerate(ranges):
        if b <= a:
            raise ValueError('Invalid range')
        if a < end and b > begin:
            edges[max(a, begin)].append((1, i, a, b, label))
            edges[min(b, end)].append((-1, i, a, b, label))
    active = {}
    result = Counter()
    previous = begin
    for t in sorted(edges):
        ordered = sorted(active.values(), key=lambda r: (r[0], -r[1]))
        for outer, inner in zip(ordered, ordered[1:]):
            if inner[1] > outer[1]:
                raise ValueError('Non-nested ranges on one thread')
        label = ordered[-1][2] if ordered else 'outside_instrumented_sections'
        result[label] += t - previous
        for sign, i, a, b, label in edges[t]:
            if sign < 0:
                del active[i]
            else:
                active[i] = (a, b, label)
        previous = t
    assert not active and sum(result.values()) == end - begin
    return {k:v for k,v in result.items() if v}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def analyze(root, streams):
    report_path = root / ('s%d-report/report.json' % streams)
    profile_path = root / ('s%d-report/profile.json' % streams)
    sqlite_path = root / ('s%d.sqlite' % streams)
    report = json.loads(report_path.read_text())
    profile = json.loads(profile_path.read_text())
    assert report['status'] == report['checks']['status'] == 'pass'
    assert report['mode'] == 'profile' and report['streams'] == streams
    assert report['delivered_events'] == report['checks']['events'] == 10000
    assert report['batch_size'] == 7 and report['bins'] == 100
    assert report['admission']['bulk_target_bytes'] == 524288
    assert report['admission']['framework_budget_gib'] == 4
    assert report['admission_pressure_trims'] == {'warmup': 0, 'steady': 0}
    assert report['bank_events'] == ([10000] if streams == 1 else [5000, 5000])
    callbacks = report['callbacks']
    warm = [r for r in callbacks if r['phase'] == 'warmup']
    steady = [r for r in callbacks if r['phase'] == 'steady']
    assert len(callbacks) == 1500 and len(warm) == 150 and len(steady) == 1350
    assert sum(r['size'] for r in warm) == 1000 and sum(r['size'] for r in steady) == 9000
    known = {r['stage'] for r in profile['cpu']}
    with sqlite3.connect(sqlite_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        strings = dict(db.execute('select id,value from StringIds'))
        nvtx = [(a, b, text if text is not None else strings.get(tid), thread)
                for a, b, text, tid, thread in db.execute(
                    'select start,end,text,textId,globalTid from NVTX_EVENTS where end>start')]
        calls = sorted(r for r in nvtx if r[2] == 'expand_calibrate_call')
        assert len(calls) == len(callbacks) and len({r[3] for r in calls}) == 1
        main_thread = calls[0][3]
        main = [r for r in nvtx if r[3] == main_thread and r[2] in known]
        begin = calls[len(warm)][0]
        end = max(r[1] for r in main)
        # This envelope starts just after the native steady timer and ends before
        # its final explicit join/finish. Bound the mismatch instead of equating them.
        wall = report['window']['completed_window_seconds']
        assert abs((end-begin)/1e9 - wall) < 0.1
        cpu = partition([(a,b,n) for a,b,n,_ in main], begin, end)
        kernels = [(a,b,s,strings[n]) for a,b,s,n in db.execute(
            'select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL')]
        devices = list(db.execute('select distinct deviceId,contextId,globalPid from CUPTI_ACTIVITY_KIND_KERNEL'))
        assert len(devices) == 1
        task_names = {'expand10','calibrate','accept_events','accumulate'}
        selected = [r for r in kernels if r[3] in task_names and r[0] >= begin and r[1] <= end]
        assert Counter(r[3] for r in selected) == Counter({n:1350 for n in task_names})
        assert len({r[2] for r in selected}) == streams
        def clipped(rows):
            return [(max(a,begin),min(b,end)) for a,b,*_ in rows if a<end and b>begin]
        groups = defaultdict(list)
        for a,b,s,n in kernels:
            if a<end and b>begin:
                groups[n].append((max(a,begin),min(b,end)))
        gpu = [dict(name=n, count=len(v), summed_seconds=sum(b-a for a,b in v)/1e9,
                    union_seconds=union_ns(v)/1e9) for n,v in groups.items()]
        copies = list(db.execute('select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'))
        copy_groups = {}
        for kind in {r[2] for r in copies}:
            rows = [r for r in copies if r[2] == kind and r[0]<end and r[1]>begin]
            copy_groups[str(kind)] = dict(count=len(rows), union_seconds=union_ns(clipped(rows))/1e9,
                summed_seconds=sum(b-a for a,b in clipped(rows))/1e9)
        reads = [r for r in nvtx if r[2] == 'Buffered pread']
        activity = dict(all_kernels_union_seconds=union_ns(clipped(kernels))/1e9,
                        task_kernels_union_seconds=union_ns(clipped(selected))/1e9,
                        buffered_pread_worker_union_seconds=union_ns(clipped(reads))/1e9,
                        copies_by_cupti_kind=copy_groups)
    return dict(streams=streams, host=report['host'], job=report['job'],
        status='pass', native_window=report['window'], envelope_seconds=(end-begin)/1e9,
        envelope_start_ns=begin, envelope_end_ns=end,
        envelope_method='First steady expand_calibrate_call NVTX start through last instrumented main-thread range end; excludes a small steady-window head/tail.',
        main_thread=main_thread, cpu_exclusive_seconds={n:t/1e9 for n,t in sorted(cpu.items(),key=lambda x:-x[1])},
        cpu_inclusive_steady=[r for r in profile['cpu'] if r['phase']=='steady'],
        gpu_kernels=sorted(gpu,key=lambda r:-r['summed_seconds']), activity=activity,
        interpretation='CPU exclusive sections partition host wall time (including blocking), not CPU utilization. GPU/copy/worker times overlap this wall time and each other; do not add them. Profile instrumentation perturbs timings. Buffered pread includes cache/OS effects, not physical Weka service time.',
        artifacts={str(p):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in
                   (report_path,profile_path,sqlite_path,root/('s%d.nsys-rep'%streams))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(args.root, s) for s in (1,2)]
    assert len({(r['host'],r['job']) for r in results}) == 1
    with args.output.open('x') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
