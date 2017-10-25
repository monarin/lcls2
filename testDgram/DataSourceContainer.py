#!/usr/bin/env python
#

import sys, os
import time
import getopt
#import pprint

sys.path.append('../build/xtcdata')
import dgram

import gc
#

class DataSource:
    def __init__(self, xtcdata_filename, verbose=0, debug=0):
        self.fd = os.open(xtcdata_filename,
                          os.O_RDONLY|os.O_LARGEFILE)
        self.config = dgram.Dgram(self.fd,
                                  verbose=verbose,
                                  debug=debug)

    def __iter__(self):
        return self

    def __next__(self):
        d=dgram.Dgram(self.fd, self.config,
                      verbose=self.verbose,
                      debug=self.debug)
        for key in sorted(d.__dict__.keys()):
            setattr(self, key, getattr(d, key))
        return self

    def get_verbose(self):
        return getattr(self.config, "verbose")
    def set_verbose(self, value):
        setattr(self.config, "verbose", value)
    verbose = property(get_verbose, set_verbose)

    def get_debug(self):
        return getattr(self.config, "debug")
    def set_debug(self, value):
        setattr(self.config, "debug", value)
    debug = property(get_debug, set_debug)


def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:f:')
    verbose=0
    debug=0
    xtcdata_filename="data.xtc"
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-v': verbose+=1
        if option=='-d': debug = int(parameter)
        if option=='-f': xtcdata_filename = parameter
    if xtcdata_filename is None:
        xtcdata_filename="data.xtc"
    if verbose>0:
        sys.stdout.write("xtcdata filename: %s\n" % xtcdata_filename)
        sys.stdout.write("verbose: %d\n" % verbose)
        sys.stdout.write("debug: %d\n" % debug)
    elif debug>0:
        sys.stdout.write("debug: %d\n" % debug)
    return (args_proper, xtcdata_filename, verbose, debug)

def getMemSize():
    pid=os.getpid()
    ppid=os.getppid()
    cmd="/usr/bin/ps -q %d --no-headers -eo size" % pid
    p=os.popen(cmd)
    size=int(p.read())
    return size

def main():
    args_proper, xtcdata_filename, verbose, debug = parse_command_line()
    size0=getMemSize()
    ds=DataSource(xtcdata_filename, verbose=verbose, debug=debug)
    for evt in ds:
        a0=evt.array0_pgp
        del(evt)
        del(a0)
        gc.collect()
        time.sleep(1)
    del(ds)
    gc.collect()
    time.sleep(2)
    size1=getMemSize()
    print("size=%d; Delta size=%d" % (size1, size1-size0))
    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h] [-v] [-d <DEBUG_NUMBER>]\n" % s)
    sys.stdout.write("%s [-f xtcdata_filename]\n" % (" "*len(s)))
    sys.exit(1)

if __name__=='__main__':
    main()

