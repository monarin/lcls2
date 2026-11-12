#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# This file is part of the 'Camera link gateway'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'Camera link gateway', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os
import sys
import time
import argparse
import importlib
import rogue

if __name__ == "__main__":

#################################################################

    # Set the argument parser
    parser = argparse.ArgumentParser()

    # Convert str to bool
    argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

    # Convert arg to a dictionary
    class keyvalue(argparse.Action):
        def __call__( self , parser, namespace,values, option_string = None):
            setattr(namespace, self.dest, dict())
            for value in values:
                # split it into key and value
                key, value = value.split('=')
                # Convert the key from str to int
                key = int(key)
                # assign into dictionary
                getattr(namespace, self.dest)[key] = value

    # Add arguments
    parser.add_argument(
        "--laneConfig",
        nargs    = '*',
        action   = keyvalue,
        required = True,
        help     = "Define the dictionary for the lane configuration. E.g. --laneConfig 0=Opal1000 1=Opal1000",
    )

    parser.add_argument(
        "--dev",
        type     = str,
        required = False,
        default  = '/dev/datadev_0',
        help     = "path to device",
    )

    parser.add_argument(
        "--pgp4",
        type     = argBool,
        required = True,
        help     = "False = PGP2b, True = PGPv4",
    )

    parser.add_argument(
        "--enLclsI",
        type     = argBool,
        required = False,
        default  = True, # Default: Enable LCLS-I hardware registers
        help     = "Enable LCLS-I hardware registers",
    )

    parser.add_argument(
        "--enLclsII",
        type     = argBool,
        required = False,
        default  = False, # Default: Disable LCLS-II hardware registers
        help     = "Enable LCLS-II hardware registers",
    )

    parser.add_argument(
        "--startupMode",
        type     = argBool,
        required = False,
        default  = False, # Default: False = LCLS-I timing mode
        help     = "False = LCLS-I timing mode, True = LCLS-II timing mode",
    )

    parser.add_argument(
        "--standAloneMode",
        type     = argBool,
        required = False,
        default  = False, # Default: False = using fiber timing
        help     = "False = using fiber timing, True = locally generated timing",
    )

    parser.add_argument(
        "--enableConfig",
        type     = argBool,
        required = False,
        default  = True,
        help     = "False to bypass the YAML load",
    )

    parser.add_argument(
        "--enableDump",
        type     = argBool,
        required = False,
        default  = False,
        help     = "True to dump .h and pre/post state",
    )

    parser.add_argument(
        "--pollEn",
        type     = argBool,
        required = False,
        default  = True,
        help     = "Enable auto-polling",
    )

    parser.add_argument(
        "--initRead",
        type     = argBool,
        required = False,
        default  = True,
        help     = "Enable read all variables at start",
    )

    parser.add_argument(
        "--seuDumpDir",
        type     = str,
        required = False,
        default  = None,
        help     = "path to SEU dump directory",
    )

    parser.add_argument(
        "--pcieBoardType",
        type     = str,
        required = False,
        default  = None,
        help     = "define the type of PCIe card, used to select I2C mapping. Options: [none or SlacPgpCardG4 or Kcu1500]",
    )

    parser.add_argument(
        "--serverPort",
        type     = int,
        required = False,
        default  = 9099,
        help     = "Zeromq server port",
    )

    parser.add_argument(
        "--releaseZip",
        type     = str,
        required = False,
        default  = None,
        help     = "Sets the default YAML configuration file to be loaded at the root.start()",
    )

    parser.add_argument(
        "--enVcMask",
        type     = lambda x: int(x,0), # "auto_int" function for hex arguments
        required = False,
        default  = 0xD,
        help     = "4-bit bitmask for selecting which VC's will be used",
    )

    parser.add_argument(
        "--guiType",
        type     = str,
        required = False,
        default  = 'PyDM',
        help     = "Sets the GUI type (PyDM or None)",
    )

    # Get the arguments
    args = parser.parse_args()

    #################################################################

    # First see if submodule packages are already in the python path
    try:
        import axi_pcie_core
        import lcls2_pgp_fw_lib
        import lcls_timing_core
        import l2si_core
        import clink_gateway_fw_lib
        import surf

    # Otherwise assume it is relative in a standard development directory structure
    except:

        # Check for release zip file path
        if args.releaseZip is not None:
            pyrogue.addLibraryPath(args.releaseZip + '/python')
        else:
            import setupLibPaths

    # Load the cameralink-gateway package
    import cameralink_gateway

    #################################################################

    print ( f'laneConfig={args.laneConfig}' )

    with cameralink_gateway.ClinkDevRoot(
            dev            = args.dev,
            pollEn         = args.pollEn,
            initRead       = args.initRead,
            laneConfig     = args.laneConfig,
            enLclsI        = (args.enLclsII or not args.startupMode),
            enLclsII       = (args.enLclsII or args.startupMode),
            startupMode    = args.startupMode,
            standAloneMode = args.standAloneMode,
            enableDump     = args.enableDump,
            enableConfig   = args.enableConfig,
            pgp4           = args.pgp4,
            seuDumpDir     = args.seuDumpDir,
            pcieBoardType  = args.pcieBoardType,
            enVcMask       = args.enVcMask,
        ) as root:

        ######################
        # Development PyDM GUI
        ######################
        if (args.guiType == 'PyDM'):
            import pyrogue.pydm
            pyrogue.pydm.runPyDM(
                serverList  = root.zmqServer.address,
                sizeX = 800,
                sizeY = 1000,
            )

        #################
        # No GUI
        #################
        elif (args.guiType == 'None'):

            # Wait to be killed via Ctrl-C
            print('Running root server.  Hit Ctrl-C to exit')
            try:
                while True:
                    time.sleep(1)
            except:
                pass
            print('Stopping root server...')
            root.stop()

        ####################
        # Undefined GUI type
        ####################
        else:
            raise ValueError("Invalid GUI type (%s)" % (args.guiType) )

    #################################################################
