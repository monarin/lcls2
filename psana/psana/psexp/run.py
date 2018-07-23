from psana.psexp.node import Smd0, SmdNode, BigDataNode
from psana.dgrammanager import DgramManager
import numpy as np

def read_event(ds, event_type=None):
    """Reads an event.

    Keywords:
    ds         -- DataSource with access to the smd or xtc files
    event_type -- used to determined config updates or other
                  types of event.
    """
    rank = ds.mpi.rank
    size = ds.mpi.size
    if size == 1:
        dm = DgramManager(ds.xtc_files, configs=ds.dm.configs)
        for evt in dm: yield evt       # safe for python2
    else:
        if ds.nodetype == 'smd0':
            Smd0(ds)
        elif ds.nodetype == 'smd':
            bd_node_ids = (np.arange(size)[ds.nsmds+1:] % ds.nsmds) + 1
            smd_node = SmdNode(ds, len(bd_node_ids[bd_node_ids==rank]))
            smd_node.run_mpi()
        elif ds.nodetype == 'bd':
            smd_node_id = (rank % ds.nsmds) + 1
            bd_node = BigDataNode(ds, smd_node_id)
            for evt in bd_node.run_mpi():
                yield evt

class ConfigUpdate(object):
    """ ConfigUpdate generator """
    def __init__(self, run):
        self.run = run

    def events(self):
        for evt in read_event(self.run.ds, event_type="config"):
            yield evt

class Run(object):
    """ Run generator """
    def __init__(self, ds):
        self.ds = ds

    def events(self):
        for evt in read_event(self.ds): yield evt

    def configUpdates(self):
        for i in range(1):
            yield ConfigUpdate(self)

    def run(self):
        """ Returns integer representaion of run no.
        default: (when no run is given) is set to -1"""
        return self.ds.run

