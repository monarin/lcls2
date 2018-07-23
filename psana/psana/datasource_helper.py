import numpy as np
import os, glob
from psana import dgram
from psana.dgrammanager import DgramManager
from mpi4py import MPI
import pickle

PERCENT_SMD = 0.25

mode = os.environ.get('PS_PARALLEL', 'mpi')
MPI = None
legion = None
if mode == 'mpi':
    from mpi4py import MPI
    def task(fn): return fn # Nop when not using Legion
elif mode == 'legion':
    import legion
    from legion import task
else:
    raise Exception('Unrecognized value of PS_PARALLEL %s' % mode)

class DataSourceHelper(object):

    def __init__(self, expstr, ds):
        self.expstr = expstr
        self.xtc_nbytes = None
        self.smd_nbytes = None
        ds.xtc_files = None
        ds.smd_files = None
        ds.run = -1
        ds.calib = None
        self.ds = ds

    def assign_node_type(self):
        """ Uses mpi size to determine nodetype for a rank."""
        ds = self.ds
        rank = ds.mpi.rank
        size = ds.mpi.size
        ds.nsmds = int(os.environ.get('PS_SMD_NODES', np.ceil((size-1)*PERCENT_SMD)))
        if rank == 0:
            ds.nodetype = 'smd0'
        elif rank < self.ds.nsmds + 1:
            ds.nodetype = 'smd'
        else:
            ds.nodetype = 'bd'
    
    def run(self, job=None):
        if mode == 'mpi':
            self.run_mpi(job=job)
        elif mode == 'legion':
            DataSourceHelper.run_legion_task()

    def run_mpi(self, job=None):
        comm = self.ds.mpi.comm
        size = self.ds.mpi.size
        if size == 1:
            return

        if job == "bcast_files":
            self.ds.xtc_files = comm.bcast(self.ds.xtc_files, root=0)
            self.ds.smd_files = comm.bcast(self.ds.smd_files, root=0)
        elif job == "bcast_configs":
            self.xtc_nbytes = comm.bcast(self.xtc_nbytes, root=0) # no. of bytes is required for mpich
            for i in range(len(self.ds.xtc_files)):
                comm.Bcast([self.configs[i], self.xtc_nbytes[i], MPI.BYTE], root=0)

            self.smd_nbytes = comm.bcast(self.smd_nbytes, root=0) 
            for i in range(len(self.ds.smd_files)):
                comm.Bcast([self.smd_configs[i], self.smd_nbytes[i], MPI.BYTE], root=0)
           
            self.ds.calib = comm.bcast(self.ds.calib, root=0)

    @task
    def run_legion_task():
        pass

    def read_configs(self):
        """ Reads configs."""
        ds = self.ds
        ds.dm = DgramManager(self.ds.xtc_files)
        self.configs = ds.dm.configs
        self.xtc_nbytes = np.array([memoryview(config).shape[0] \
                for config in self.configs], dtype='i')
        
        self.smd_configs = []
        if self.ds.smd_files is not None:
            ds.smd_dm = DgramManager(self.ds.smd_files)
            self.smd_configs = ds.smd_dm.configs # FIXME: there should only be one type of config
        self.smd_nbytes = np.array([memoryview(config).shape[0] \
                for config in self.smd_configs], dtype='i')

        ds.calib = self.get_calib_dict(run_no=ds.run)
    
    def init_configs(self):
        """ Initialize configs.
        This method is only used in mpi mode when non-0 ranks get
        configs from bcast method."""
        self.configs = [dgram.Dgram() for i in range(len(self.ds.xtc_files))]
        self.smd_configs = [dgram.Dgram() for i in range(len(self.ds.smd_files))]

    def parse_expstr(self):
        run = None
        expstr = self.expstr
        # Check if we are reading file(s) or an experiment
        read_exp = False
        if isinstance(expstr, (str)):
            if expstr.find("exp") == -1:
                xtc_files = [expstr]
                smd_files = None
            else:
                read_exp = True
        elif isinstance(expstr, (list, np.ndarray)):
            xtc_files = expstr
            smd_files = None

        # Reads list of xtc files from experiment folder
        if read_exp:
            opts = expstr.split(':')
            exp = {}
            for opt in opts:
                items = opt.split('=')
                assert len(items) == 2
                exp[items[0]] = items[1]

            run = -1
            if 'dir' in exp:
                xtc_path = exp['dir']
            else:
                xtc_dir = os.environ.get('SIT_PSDM_DATA', '/reg/d/psdm')
                xtc_path = os.path.join(xtc_dir, exp['exp'][:3], exp['exp'], 'xtc')

            if 'run' in exp:
                run = int(exp['run'])

            if run > -1:
                xtc_files = glob.glob(os.path.join(xtc_path, '*r%s*.xtc'%(str(run).zfill(4))))
            else:
                xtc_files = glob.glob(os.path.join(xtc_path, '*.xtc'))

            xtc_files.sort()

            smd_dir = os.path.join(xtc_path, 'smalldata')
            smd_files = [os.path.join(smd_dir,
                                      os.path.splitext(os.path.basename(xtc_file))[0] + '.smd.xtc')
                         for xtc_file in xtc_files]

        self.ds.xtc_files = xtc_files
        self.ds.smd_files = smd_files
        if run is not None:
            self.ds.run = run

    def get_calib_dict(self, run_no=-1):
        """ Creates dictionary object that stores calibration constants.
        This routine will be replaced with calibration reading (psana2 style)"""
        calib_dir = os.environ.get('PS_CALIB_DIR')
        calib = None
        if calib_dir:
            gain_mask = None
            pedestals = None
            if os.path.exists(os.path.join(calib_dir,'gain_mask.pickle')):
                gain_mask = pickle.load(open(os.path.join(calib_dir,'gain_mask.pickle'), 'r'))

            # Find corresponding pedestals
            if run_no > -1: # Do not fetch pedestals when run_no is not given
                if os.path.exists(os.path.join(calib_dir,'pedestals.npy')):
                    pedestals = np.load(os.path.join(calib_dir,'pedestals.npy'))
                else:
                    files = glob.glob(os.path.join(calib_dir,"*-end.npy"))
                    darks = np.sort([int(os.path.basename(file_name).split('-')[0]) for file_name in files])
                    sel_darks = darks[(darks < run_no)]
                    if sel_darks.size > 0:
                        if os.path.exists(os.path.join(calib_dir,'%s-end.npy'%sel_darks[0])):
                            pedestals = np.load(os.path.join(calib_dir, '%s-end.npy'%sel_darks[0]))
            calib = {'gain_mask': gain_mask,
                     'pedestals': pedestals}
        return calib



