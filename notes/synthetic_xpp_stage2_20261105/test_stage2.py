"""Negative mutation checks, transport guard and frozen production schedule checks."""
import json
from pathlib import Path
import struct
import tempfile
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import requests

from fixture import ROOT, SNAPSHOT, frozen_report, plan, repeat_schedule, rewrite, wire
from verify import compare_record
from calibration import Guard, ALLOWED, body_hash
import calibration


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


class DeploymentTests(unittest.TestCase):
    def exercise(self, failure=None):
        """Exercise the actual deploy loop; replace only external dependencies."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory);snapshot=root/'snapshot';snapshot.mkdir()
            execution=root/'deployment';plan_path=root/'plan.json'
            items=[]
            for n in range(2):
                raw=f'payload-{n}'.encode();name=f'{n}.bin';(snapshot/name).write_bytes(raw)
                items.append(dict(file=name,sha256=wire.sha(raw),document={'ctype':f'type{n}','id_data':calibration.PLACEHOLDER}))
            plan_data=dict(items=items,existing_before={},protected_before={},
                script_sha256=wire.file_sha(calibration.HERE/'calibration.py'))
            calibration.save(plan_path,plan_data)
            for key,value in dict(ROOT=root,SNAPSHOT=snapshot,EXECUTION=execution,PLAN=plan_path).items():
                stack.enter_context(patch.object(calibration,key,value))
            for name,value in (('destination',{}),('protected',{}),('check_existing',None),
                               ('check_run_slots',None),('get',{'status':True})):
                stack.enter_context(patch.object(calibration,name,return_value=value))
            auth=stack.enter_context(patch('krtc.KerberosTicket'))
            auth.return_value.getAuthHeaders.side_effect=lambda:{'Authorization':'unit-test-only'}
            calls=[]
            def transport(session,method,url,**kwargs):
                index=len(calls);n=index//2;kind='data' if index%2==0 else 'document'
                prefix=execution/f'{n:02d}-{kind}'
                intent=json.loads(prefix.with_suffix('.intent.json').read_text())
                self.assertEqual(intent['body_sha256'],body_hash(kwargs))
                self.assertEqual(method,'POST');self.assertFalse(kwargs['allow_redirects'])
                self.assertEqual(url,calibration.RW+calibration.DB+('/gridfs/' if kind=='data' else '/'+calibration.COLLECTION+'/'))
                if kind=='data':self.assertEqual(kwargs['headers']['Content-Type'],'application/octet-stream')
                else:self.assertEqual(kwargs['json']['id_data'],f'id-{index-1}')
                calls.append(kind)
                response=requests.Response();response.status_code=200
                response._content=json.dumps({'_id':f'id-{index}'}).encode()
                response.headers['Content-Type']='application/json'
                response.headers['Set-Cookie']='must-not-enter-git-evidence'
                if index==1 and failure:
                    response._content=b'<html>uncertain server result</html>'
                    response.status_code={'http':500,'redirect':302,'json':200}[failure]
                    if failure=='redirect':response.headers['Location']='https://example.invalid/'
                return response
            stack.enter_context(patch.object(requests.sessions.Session,'request',autospec=True,side_effect=transport))
            guard=Guard();guard.install()
            if failure:
                with self.assertRaises((ValueError,requests.exceptions.JSONDecodeError)):
                    calibration.deploy(guard,wire.file_sha(plan_path))
                self.assertEqual(calls,['data','document'])
                self.assertFalse((execution/'journal.json').exists())
                self.assertTrue((execution/'00-data.response.json').exists())
                self.assertTrue((execution/'00-document.intent.json').exists())
                self.assertFalse((execution/'00-document.response.json').exists())
                saved=json.loads((execution/'00-document.http.json').read_text())
                self.assertEqual(Path(saved['body_path']).read_bytes(),b'<html>uncertain server result</html>')
                self.assertNotIn('Set-Cookie',saved['headers'])
                with self.assertRaises(FileExistsError):calibration.deploy(guard,wire.file_sha(plan_path))
                self.assertEqual(len(calls),2)  # No retry and no later type was uploaded.
            else:
                calibration.deploy(guard,wire.file_sha(plan_path))
                journal=json.loads((execution/'journal.json').read_text())
                self.assertEqual([(r['id_data'],r['id_doc']) for r in journal],[('id-0','id-1'),('id-2','id-3')])
                self.assertEqual(auth.return_value.getAuthHeaders.call_count,5)
                self.assertEqual(len(list(execution.glob('*.intent.json'))),4)

    def test_success_intent_before_post_and_allocated_data_ids(self):self.exercise()
    def test_http_failure_stops_and_preserves_orphan_evidence(self):self.exercise('http')
    def test_json_failure_stops_and_preserves_body(self):self.exercise('json')
    def test_redirect_stops_without_following(self):self.exercise('redirect')


if __name__=='__main__':unittest.main()
