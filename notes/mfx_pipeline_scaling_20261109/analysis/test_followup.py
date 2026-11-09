import copy
import unittest
from followup_audit import gate
from test_audit import example

class Followup(unittest.TestCase):
    def test_cold_valid(self):gate(example()['measurement'],'cold')
    def test_warm_gate(self):
        with self.assertRaises(ValueError):gate(example()['measurement'],'warm')
    def test_bad_nic_delta(self):
        m=example()['measurement'];m['nic_delta']['nic']['rx_bytes']+=1
        with self.assertRaises(ValueError):gate(m,'cold')
    def test_device_change(self):
        m=example()['measurement'];m['nic_after']['nic']['device']='other'
        with self.assertRaises(ValueError):gate(m,'cold')
    def test_missing_cache_file(self):
        m=example()['measurement'];m['conditioning']['cache']['files'].pop()
        with self.assertRaises(ValueError):gate(m,'cold')
