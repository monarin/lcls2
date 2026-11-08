"""Host fixtures exercise the actual SQLite analysis, not only interval math."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

from analyze_real import analyze


class Tests(unittest.TestCase):
    def fixture(self, root, overlap=False):
        path=Path(root)/'trace.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript('''
                create table StringIds(id integer,value text);
                create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,streamId integer,
                    deviceId integer,contextId integer,globalPid integer,demangledName integer,correlationId integer);
                create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,nameId integer,correlationId integer);
            ''')
            names=['gather_canonical_u16','calibrate','accept_events','accumulate','cuLaunchKernel']
            db.executemany('insert into StringIds values (?,?)',enumerate(names))
            for stream,start in [(13,10),(14,25 if overlap else 60)]:
                for i in range(4):
                    corr=stream*10+i
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?)',
                               (start+i*10,start+i*10+8,stream,0,1,123, i,corr))
                    db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?)',
                               (start+i*10-2,start+i*10-1,4,corr))
        return path,dict(status='pass',streams=2,callbacks=[{},{}])

    def test_real_gather_handoffs_overlap_and_serial(self):
        for overlap in (False,True):
            with tempfile.TemporaryDirectory() as root:
                p,r=self.fixture(root,overlap)
                got=analyze(p,r)
                self.assertEqual(got['submission_handoffs']['next_launch_before_previous_sum_end'],int(overlap))
                self.assertEqual(got['task_kernels']['cross_stream_overlap_ns']>0,overlap)

    def test_incomplete_kernels_wrong_context_and_ambiguous_launch_rejected(self):
        mutations=['delete from CUPTI_ACTIVITY_KIND_KERNEL where demangledName=0',
                   'update CUPTI_ACTIVITY_KIND_KERNEL set contextId=2 where streamId=14',
                   'insert into CUPTI_ACTIVITY_KIND_RUNTIME select * from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=130']
        for sql in mutations:
            with tempfile.TemporaryDirectory() as root:
                p,r=self.fixture(root)
                with sqlite3.connect(p) as db:db.execute(sql)
                with self.assertRaises(ValueError):analyze(p,r)


if __name__=='__main__':unittest.main()
