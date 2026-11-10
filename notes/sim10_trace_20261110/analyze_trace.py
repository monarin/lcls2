"""Read-only analysis of Nsight SQLite exports; times are trace-origin seconds."""
import bisect
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(b-a for a,b in intervals)


def clipped(intervals, start, end):
    return [(max(a,start),min(b,end)) for a,b in intervals if a < end and b > start]


def analyze(path):
    conn = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    rows = lambda q: [dict(x) for x in conn.execute(q)]
    nvtx = rows("""SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) label
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL AND coalesce(n.text,s.value) IN
        ('psana.bd.loop','psana.group.issue','psana.input.poll','psana.group.submit')
        ORDER BY n.start""")
    loop = [r for r in nvtx if r['label']=='psana.bd.loop']
    assert len(loop)==1
    start,end = loop[0]['start'],loop[0]['end']
    groups=[r for r in nvtx if r['label']=='psana.group.issue']
    polls=[r for r in nvtx if r['label']=='psana.input.poll']
    kernels=rows('''SELECT k.start,k.end,k.streamId,k.correlationId,s.value name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        ORDER BY k.start''')
    copies=rows('SELECT start,end,bytes,streamId,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')
    gpu_union=union([(r['start'],r['end']) for r in kernels+copies])
    kernel_union=union([(r['start'],r['end']) for r in kernels])
    copy_union=union([(r['start'],r['end']) for r in copies])
    samples=[]
    assert len(groups)==100
    for index in (20,21,50):
        g=groups[index];a,b=g['start'],g['end']
        ps=[r for r in polls if a<=r['start'] and r['end']<=b and r['globalTid']==g['globalTid']]
        following=[r for r in kernels if r['start']>=b][:2]
        samples.append(dict(batch_index=index,start_s=a/1e9,end_s=b/1e9,
            duration_ms=(b-a)/1e6,poll_count=len(ps),poll_ms=sum(r['end']-r['start'] for r in ps)/1e6,
            gpu_kernel_or_copy_union_ms=duration(clipped(gpu_union,a,b))/1e6,
            gpu_kernel_union_ms=duration(clipped(kernel_union,a,b))/1e6,
            following_kernels=[dict(name=r['name'],stream=r['streamId'],start_s=r['start']/1e9,end_s=r['end']/1e9) for r in following]))
    out=dict(sqlite=str(path),loop_start_s=start/1e9,loop_end_s=end/1e9,loop_seconds=(end-start)/1e9,
        gpu_kernel_or_copy_union_seconds=duration(clipped(gpu_union,start,end))/1e9,
        gpu_kernel_union_seconds=duration(clipped(kernel_union,start,end))/1e9,
        gpu_copy_union_seconds=duration(clipped(copy_union,start,end))/1e9,
        kernel_summary=rows('''SELECT s.value name,k.streamId,count(*) count,sum(k.end-k.start)/1e9 summed_seconds
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id GROUP BY k.shortName,k.streamId'''),
        copy_summary=rows('''SELECT e.label kind,count(*) count,sum(m.bytes) bytes,sum(m.end-m.start)/1e9 summed_seconds
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id GROUP BY m.copyKind'''),
        cuda_api_summary=rows('''SELECT s.value name,count(*) count,sum(r.end-r.start)/1e9 summed_seconds
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id GROUP BY r.nameId ORDER BY summed_seconds DESC LIMIT 12'''),
        samples=samples,
        limitations=['CUDA API sums include multiple host threads and may overlap.',
            'GPU activity union covers traced kernels and memcpy, not hardware utilization or network I/O.',
            'Nsight significantly perturbs throughput; use unprofiled results for rates.',
            'Sample indices are zero-based group.issue batches; timestamps use Nsight trace origin.'])
    conn.close()
    return out


if __name__ == '__main__':
    print(json.dumps([analyze(Path(p)) for p in sys.argv[1:]],indent=2))
