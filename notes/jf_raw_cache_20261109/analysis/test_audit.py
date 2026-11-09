import copy,unittest
from audit import gate

def record():
    files=[dict(path=str(i),pages=10,resident_pages=0) for i in range(5)]
    return dict(conditioning=dict(kind='cold',cache=dict(files=files,pages=50,resident_pages=0)),loop_seconds=2,nic_before=dict(p=dict(device='d',monotonic=1,rx_bytes_phy=0,tx_bytes_phy=0)),nic_after=dict(p=dict(device='d',monotonic=4,rx_bytes_phy=30,tx_bytes_phy=1)),nic_delta=dict(p=dict(seconds=3,rx_bytes=30,tx_bytes=1,rx_gib_s=30/(1<<30)/3)))
class Audit(unittest.TestCase):
    def test_valid(self):gate(record(),'cold')
    def test_cold(self):
        r=record();r['conditioning']['cache']['files'][0]['resident_pages']=1;r['conditioning']['cache']['resident_pages']=1
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_warm_per_file(self):
        r=record()
        with self.assertRaises(ValueError):gate(r,'warm')
    def test_wrong_kind(self):
        r=record();r['conditioning']['kind']='warm'
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_reset(self):
        r=record();r['nic_after']['p']['rx_bytes_phy']=-1
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_bracket(self):
        r=record();r['loop_seconds']=4
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_duplicate_file(self):
        r=record();r['conditioning']['cache']['files'][0]['path']='1'
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_bad_delta(self):
        r=record();r['nic_delta']['p']['rx_bytes']=31
        with self.assertRaises(ValueError):gate(r,'cold')
