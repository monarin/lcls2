"""Read the two NVTX-delimited, fully joined phases from the native capture."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'real_jf_concurrency_20261108'))
from analyze_real import intervals


def analyze(path,report):
    assert report['status']=='pass'
    bd,=[r for r in report['rows'] if r['role']=='bd']
    assert len(bd['timestamps'])==64 and len(bd['callbacks'])==16
    assert all(s['sizes']==[4]*4 and s['depth']==2 for s in bd['splits'])
    assert bd['pressure_drains']==0
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        names=dict(db.execute('select id,value from StringIds'))
        records=list(db.execute('select start,end,streamId,demangledName,correlationId,globalPid,deviceId,contextId from CUPTI_ACTIVITY_KIND_KERNEL'))
        assert len({tuple(r[5:]) for r in records})==1,'Expected one CUDA process/device/context'
        phases={}
        for label,count,first in (('PIPELINE_64_EVENTS',16,'gather_canonical_u16'),('RESIDENT_CONTROL_8_EVENTS',2,'calibrate')):
            span,=list(db.execute('select start,end,globalTid from NVTX_EVENTS where text=?',(label,)))
            a,b,tid=span
            assert a<b
            # Both ranges end after explicit stream joins; no selected work may straddle a boundary.
            rows=[r for r in records if a<=r[0] and r[1]<=b]
            selected_names={'calibrate','accept_events','accumulate'}|({first} if first!='calibrate' else set())
            selected=[r for r in rows if names[r[3]].split('(')[0].strip() in selected_names]
            counts=Counter(names[r[3]].split('(')[0].strip() for r in selected)
            assert counts==Counter({n:count for n in selected_names}),(label,counts)
            streams=sorted({r[2] for r in selected});assert len(streams)==2
            assert sum(names[r[3]].split('(')[0].strip() in selected_names and r[0]<b and r[1]>a for r in records)==len(selected)
            executions=[]
            for stream in streams:
                starts=sorted((r for r in selected if r[2]==stream and names[r[3]].split('(')[0].strip()==first))
                sums=sorted((r for r in selected if r[2]==stream and names[r[3]].split('(')[0].strip()=='accumulate'))
                assert len(starts)==len(sums)==count//2
                for start,end in zip(starts,sums):
                    launch,=list(db.execute('select start,end,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=?',(start[4],)))
                    assert launch[2]==tid and start[0]<end[1]
                    executions.append(dict(stream=stream,launch_start=launch[0],sum_end=end[1]))
            executions.sort(key=lambda e:e['launch_start'])
            gaps=[q['launch_start']-p['sum_end'] for p,q in zip(executions,executions[1:])]
            phases[label]=dict(start_ns=a,end_ns=b,streams=streams,kernel_counts=dict(counts),
                selected=intervals([(r[0],r[1],r[2]) for r in selected]),
                all_kernels=intervals([(r[0],r[1],r[2]) for r in rows]),
                handoffs=dict(gaps_ns=gaps,next_launch_before_previous_sum_end=sum(g<0 for g in gaps)))
        diagnostics=[dict(zip(('severity','text','globalPid'),r)) for r in db.execute('select severity,text,globalPid from DIAGNOSTIC_EVENT')]
        return dict(status='pass',sqlite=str(path),cuda_identity=dict(globalPid=records[0][5],device=records[0][6],context=records[0][7]),
            phases=phases,diagnostics=diagnostics,
            limitations='Instrumented short diagnostic, includes startup/warmup. Resident control bypasses I/O and dense gather; demonstrates possibility, not guaranteed or efficient overlap. No throughput or copy/compute-overlap claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path);p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.sqlite,json.loads(a.report.read_text()))
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
