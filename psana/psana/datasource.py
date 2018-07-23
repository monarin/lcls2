import sys, os, glob
from psana.detector.detector import Detector
from psana.psexp.run import Run
from psana.psexp.tools import MpiComm
from psana.datasource_helper import DataSourceHelper
from psana.dgrammanager import DgramManager

class DataSource(object):
    """ Read XTC files  """ 
    def __init__(self, expstr, filter=0, batch_size=1, max_events=0):
        """Initializes datasource.
        
        Keyword arguments:
        expstr     -- experiment string (eg. exp=xpptut13:run=1) or 
                      a file or list of files (eg. 'data.xtc' or ['data0.xtc','dataN.xtc'])
        batch_size -- length of batched offsets
        max_events -- no. of maximum events
        """
        self.filter = filter
        assert batch_size > 0
        self.batch_size = batch_size
        self.max_events = max_events
        
        self.mpi = MpiComm()
        ds_helper = DataSourceHelper(expstr, self)
        ds_helper.assign_node_type()
        
        if self.nodetype == 'smd0':
            ds_helper.parse_expstr()
        ds_helper.run(job="bcast_files")
        
        if self.nodetype == 'smd0':
            ds_helper.read_configs()
        else:
            ds_helper.init_configs()
        ds_helper.run(job="bcast_configs")

        if self.nodetype != 'smd0':
            self.smd_dm = DgramManager(self.smd_files, configs=ds_helper.smd_configs) 
            self.dm = DgramManager(self.xtc_files, configs=ds_helper.configs)
        
        if self.nodetype in ('bd', 'smd0'):
            self.Detector = Detector(self.dm.configs, calib=self.calib) 

    def runs(self): 
        nruns = 1
        for run_no in range(nruns):
            yield Run(self)

    def events(self): 
        for run in self.runs():
            for evt in run.events(): yield evt
    
    @property
    def _configs(self):
        assert len(self.dm.configs) > 0
        return self.dm.configs
    
