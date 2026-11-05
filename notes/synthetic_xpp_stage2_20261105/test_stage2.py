"""Negative mutation checks, transport guard and frozen production schedule checks."""
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import numpy as np
import requests

from fixture import ROOT, SNAPSHOT, frozen_report, plan, repeat_schedule, rewrite, wire
from verify import compare_record
from calibration import Guard, ALLOWED, body_hash


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e=frozen_report('inventory')['streams'][3]
        blob=(SNAPSHOT/'smd'/Path(e['smd']['path']).name).read_bytes()
        records=[raw for _,raw in wire.records(blob)]
        cls.schemas=wire.schemas(records[0])
        cls.event=next(raw for raw in records if wire.service(raw)==12)
        cls.begin=next(raw for raw in records if wire.service(raw)==4)

    def test_valid_offset_and_timestamp_change(self):
        ts=wire.from_ns(wire.to_ns(wire.header(self.event)[0])+1234)
        new=rewrite(self.event,ts,self.schemas,4321,2097340)
        compare_record(self.event,new,self.schemas,ts,4321,2097340)

    def test_wrong_offset_rejected(self):
        ts=wire.header(self.event)[0]
        new=rewrite(self.event,ts,self.schemas,4321,2097340)
        with self.assertRaises(ValueError):compare_record(self.event,new,self.schemas,ts,1234,2097340)

    def test_control_change_rejected(self):
        new=bytearray(self.event);new[8]^=1
        with self.assertRaises(ValueError):compare_record(self.event,bytes(new),self.schemas,wire.header(self.event)[0])

    def test_payload_change_rejected(self):
        new=bytearray(self.event);new[-1]^=1
        with self.assertRaisesRegex(ValueError,'payload change'):compare_record(self.event,bytes(new),self.schemas,wire.header(self.event)[0])

    def test_wrong_timestamp_rejected(self):
        old=wire.header(self.event)[0]
        new=rewrite(self.event,wire.from_ns(wire.to_ns(old)+100),self.schemas)
        with self.assertRaisesRegex(ValueError,'timestamp'):compare_record(self.event,new,self.schemas,old)

    def test_wrong_run_rejected(self):
        ts=wire.header(self.begin)[0];new=bytearray(rewrite(self.begin,ts,self.schemas))
        key,schema=wire.schema_for(self.schemas,'runinfo');p,_,_,_=wire.spans(new,key,schema)['runnum']
        struct.pack_into('<I',new,p,4)
        with self.assertRaises(ValueError):compare_record(self.begin,bytes(new),self.schemas,ts)


class ScheduleTests(unittest.TestCase):
    def test_both_production_mappings_match_frozen(self):
        index=np.load(SNAPSHOT/'indexes/s002.npy',allow_pickle=False)
        for run,count in ((3,2400),(4,400000)):
            rec,ts,cyc,_=repeat_schedule(index,count,151152)
            with np.load(SNAPSHOT/f'run{run}_mapping.npz',allow_pickle=False) as frozen:
                for key,value in (('source_record',rec),('timestamp',ts),('cycle',cyc)):
                    np.testing.assert_array_equal(value,frozen[key])

    def test_seams_rollover_and_truncation(self):
        index=np.load(SNAPSHOT/'indexes/s002.npy',allow_pickle=False);p=plan(index,'seam')
        mask=index['service'][p['source_record']]==12
        self.assertEqual(p['cycle'][mask].tolist(),[0,0,0,1,1,1,2,2])
        self.assertTrue(np.all(p['timestamp'][1:]>p['timestamp'][:-1]))
        times=p['timestamp'][mask];self.assertGreater(int(times[3])>>32,int(times[2])>>32)


class GuardTests(unittest.TestCase):
    def test_only_one_predeclared_post_can_pass(self):
        with patch.object(requests.sessions.Session,'request',return_value=object()) as network:
            g=Guard();g.install();url=sorted(ALLOWED)[0];kwargs={'data':b'payload'}
            with self.assertRaises(ValueError):requests.post(url,**kwargs)
            g.pending=(url,body_hash(kwargs))
            with self.assertRaises(ValueError):requests.post(url,data=b'changed')
            requests.post(url,**kwargs)
            with self.assertRaises(ValueError):requests.post(url,**kwargs)
            with self.assertRaises(ValueError):requests.delete(url)
            with self.assertRaises(ValueError):requests.post(url.replace('cdb_prjdatpsana','cdb_jungfrau_000010'),**kwargs)
            self.assertEqual(network.call_count,1)


if __name__=='__main__':unittest.main()
