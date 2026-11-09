import copy
import unittest
from audit import check


def example():
    counts=[1,1,1]+[0]*97
    cache=dict(pages=100,resident_pages=0,files=[dict(path=str(i),pages=10,resident_pages=0) for i in range(10)])
    before={'nic':dict(monotonic=1,device='pci0',rx_bytes_phy=10,tx_bytes_phy=10)}
    after={'nic':dict(monotonic=13,device='pci0',rx_bytes_phy=10010,tx_bytes_phy=10)}
    row=dict(role='bd',rank=2,host='node',job='job',cpu_affinity=[2,3,4,5],timestamps=[100,200,300],iteration_seconds=9,
        runtime=dict(device_pci='gpu0',device_mask='uuid0',compat_mode='ON',effective=dict(num_threads=4,task_size=4<<20,bounce_buffer_size=16<<20,auto_direct_io_read=False)),
        io=dict(requested_bytes=3358200*3,completed_bytes=3358200*3,requests=15,completed_requests=15,pressure_drains=0),
        checks=dict(status='pass',events=3,counts=counts,banks=[dict(counts=counts)]),
        callbacks=[dict(size=3,timestamps=[100,200,300],stream=42)])
    return dict(status='pass',settings=dict(batch=7,bds=1,streams=1,task='full',cache='cold',diagnostic=False),
        events=3,jf_bytes=3358200*3,events_per_second=.3,jf_gib_s=3358200*3/(1<<30)/10,
        rows=[dict(role='smd0',host='node',job='job',cpu_affinity=[0]),dict(role='eb',host='node',job='job',cpu_affinity=[1]),row],
        measurement=dict(loop_seconds=10,conditioning=dict(cache=cache),cache_after=cache,
            nic_before=before,nic_after=after,idle=dict(before=before,after=after),
            nic_delta={'nic':dict(seconds=12,rx_bytes=10000,tx_bytes=0,rx_gib_s=10000/(1<<30)/12)}))

class Audit(unittest.TestCase):
    def test_valid(self):self.assertEqual(len(check(example())),1)
    def test_duplicate_event(self):
        r=example();r['rows'][2]['timestamps'][2]=200
        with self.assertRaises(ValueError):check(r)
    def test_wrong_bank_counts(self):
        r=example();r['rows'][2]['checks']['banks'][0]['counts']=list(reversed(r['rows'][2]['checks']['counts']))
        with self.assertRaises(ValueError):check(r)
    def test_incomplete_read(self):
        r=example();r['rows'][2]['io']['completed_bytes']-=1
        with self.assertRaises(ValueError):check(r)
    def test_cold_gate(self):
        r=example();r['measurement']['conditioning']['cache']['resident_pages']=1
        r['measurement']['conditioning']['cache']['files'][0]['resident_pages']=1
        with self.assertRaises(ValueError):check(r)
    def test_counter_reset(self):
        r=example();r['measurement']['nic_after']['nic']['rx_bytes_phy']=0
        with self.assertRaises(ValueError):check(r)
    def test_overlap_cpu_affinity(self):
        r=example();r['rows'][2]['cpu_affinity'][0]=0
        with self.assertRaises(ValueError):check(r)
    def test_false_rate(self):
        r=example();r['jf_gib_s']*=2
        with self.assertRaises(ValueError):check(r)
    def test_exclusive_timing_overflow(self):
        r=example();r['settings']['diagnostic']=True
        r['rows'][2]['profile']={'cpu':{'wait':dict(wall_exclusive=20,thread_exclusive=0)}}
        with self.assertRaises(ValueError):check(r)
