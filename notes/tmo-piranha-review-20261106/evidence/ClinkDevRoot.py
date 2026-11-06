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
import pyrogue as pr
import rogue
import click

import cameralink_gateway as clDev
import axipcie

import lcls2_pgp_fw_lib.shared as shared

import ClinkFeb               as feb
import surf.protocols.batcher as batcher
import surf.protocols.clink   as cl
import l2si_core              as l2si

rogue.Version.minVersion('6.1.1')
# rogue.Version.exactVersion('5.1.0')

class DummyBiDirStream(rogue.interfaces.stream.Master, rogue.interfaces.stream.Slave):

    def __init__(self):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

    def __eq__(self,other):
        pyrogue.streamConnectBiDir(other,self)

class ClinkDevRoot(shared.Root):

    def __init__(self,
                 dataDebug      = False,
                 dev            = '/dev/datadev_0',# path to PCIe device
                 enLclsI        = True,
                 enLclsII       = False,
                 startupMode    = False, # False = LCLS-I timing mode, True = LCLS-II timing mode
                 standAloneMode = False, # False = using fiber timing, True = locally generated timing
                 pgp4           = False, # true = PGPv4, false = PGP2b
                 pollEn         = True,  # Enable automatic polling registers
                 initRead       = True,  # Read all registers at start of the system
                 laneConfig     = {0: 'Opal1000'},
                 seuDumpDir     = None,
                 enableDump     = False,
                 enableConfig   = True,
                 pcieBoardType  = None,
                 enVcMask       = 0xD, # Enable lane mask: Don't connect data stream (VC1) by default because intended for C++ process
                 zmqSrvEn       = True,  # Flag to include the ZMQ server
                 **kwargs):

        # Set the FEB firmware Version lock = https://github.com/slaclab/cameralink-gateway/blob/master/firmware/targets/shared_version.mk
        self.FebVersionLock = 0x08020000

        # Set the FEB firmware Version lock = https://github.com/slaclab/lcls2-pgp-pcie-apps/blob/master/firmware/targets/shared_config.mk
        self.PcieVersionLock = 0x03060000

        # Set local variables
        self.laneConfig     = laneConfig
        self.dev            = dev
        self.startupMode    = startupMode
        self.standAloneMode = standAloneMode
        self.enableDump     = enableDump
        self.enableConfig   = enableConfig
        self.defaultFile    = []

        # Generate a list of configurations
        for lane in self.laneConfig:
            self.defaultFile.append(f'config/{self.laneConfig[lane]}/lane{lane}.yml')

        # Pass custom value to parent via super function
        super().__init__(
            dev         = dev,
            pgp4        = pgp4,
            pollEn      = pollEn,
            initRead    = initRead,
            **kwargs)

        # Add ZMQ server
        if zmqSrvEn:
            self.zmqServer = pr.interfaces.ZmqServer(root=self, addr='*', port=0)
            self.addInterface(self.zmqServer)

        # Check for simulation
        if dev == 'sim':
            # Set the timeout
            self._timeout = 100000000 # firmware simulation slow and timeout base on real time (not simulation time)

        else:
            # Set the timeout
            self._timeout = 5000000 # 5.0 seconds default
                     
        # Unhide the RemoteVariableDump command
        self.RemoteVariableDump.hidden = False

        # Create memory interface
        self.memMap = axipcie.createAxiPcieMemMap(dev, 'localhost', 8000)
        self.memMap.setName('PCIe_Bar0')

        # Instantiate the top level Device and pass it the memory map
        self.add(clDev.ClinkPcieFpga(
            name       = 'ClinkPcie',
            memBase    = self.memMap,
            laneConfig = self.laneConfig,
            pgp4       = pgp4,
            enLclsI    = enLclsI,
            enLclsII   = enLclsII,
            boardType  = pcieBoardType,
            expand     = True,
        ))

        # Create empty list
        self.dmaStreams     = [[None for x in range(4)] for y in range(4)]
        self._srp           = [None for x in range(4)]
        self._frameGen      = [None for x in range(4)]
        self.RemRxLinkReady = [None for x in range(4)]
        self.semDataWriter  = [None for x in range(4)]
        self._dbg           = [None for x in range(4)]
        self.unbatchers     = [None for x in range(4)]
        self.enVcMask       = [False for x in range(4)]

        # Create DMA streams
        for lane in self.laneConfig:

            for vc in range(4):
                if enVcMask & (0x1 << vc):
                    self.enVcMask[vc] = True
                    if (dev != 'sim'):
                        self.dmaStreams[lane][vc] = rogue.hardware.axi.AxiStreamDma(dev,(0x100*lane)+vc,1)
                    else:
                        self.dmaStreams[lane][vc] = rogue.interfaces.stream.TcpClient('localhost', (8000+2)+(512*lane)+2*vc)

        # Check if not doing simulation
        if (dev != 'sim'):

            # Create the stream interface
            for lane in self.laneConfig:

                # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
                if self.enVcMask[0]:
                    # Create the SRPv3
                    self._srp[lane] = rogue.protocols.srp.SrpV3()
                    self._srp[lane].setName(f'SRPv3[{lane}]')
                    # Connect DMA to SRPv3
                    self.dmaStreams[lane][0] == self._srp[lane]

                    # Add pointer to the list
                    self.RemRxLinkReady[lane] = self.ClinkPcie.Hsio.PgpMon[lane].RxStatus.RemRxLinkReady if pgp4 else self.ClinkPcie.Hsio.PgpMon[lane].RxRemLinkReady

                    # Check if PGP[lane].VC[2] = Camera UART (streaming data) is disabled
                    if not self.enVcMask[2]:
                        # Assign a "dummy" stream to the serial path so it doesn't break variable tree
                        self.dmaStreams[lane][2] = DummyBiDirStream()

                    # CameraLink Feb Board
                    self.add(feb.ClinkFeb(
                        name       = (f'ClinkFeb[{lane}]'),
                        memBase    = self._srp[lane],
                        serial     = self.dmaStreams[lane][2],
                        camType    = self.laneConfig[lane],
                        enableDeps = [self.RemRxLinkReady[lane]], # Only allow access if the PGP link is established
                        expand     = True,
                    ))

                # Check if PGP[lane].VC[3] = SEM UART (streaming data) is enabled
                if self.enVcMask[3]:
                    # Add SEM module
                    if seuDumpDir is not None:
                        # Create the writer
                        self.semDataWriter[lane] = feb.SemAsciiFileWriter(index=lane,dumpDir=seuDumpDir)
                        # Connect DMA to writer
                        self.dmaStreams[lane][3] >> self.semDataWriter[lane]

        # Else doing Rogue VCS simulation
        else:
            self.roguePgp = shared.RogueStreams(pgp4=pgp4)

            # Create the stream interface
            for lane in range(laneSize):

                # Check if PGP[lane].VC[1] = Camera Image (streaming data) is enabled
                if self.enVcMask[1]:

                    # Create the frame generator
                    self._frameGen[lane] = MyCustomMaster()

                    # Connect the frame generator
                    print(self.roguePgp.pgpStreams)
                    self._frameGen[lane] >> self.roguePgp.pgpStreams[lane][1]

                    # Create a command to execute the frame generator
                    self.add(pr.BaseCommand(
                        name         = f'GenFrame[{lane}]',
                        function     = lambda cmd, lane=lane: self._frameGen[lane].myFrameGen(),
                    ))

                    # Create a command to execute the frame generator. Accepts user data argument
                    self.add(pr.BaseCommand(
                        name         = f'GenUserFrame[{lane}]',
                        function     = lambda cmd, lane=lane: self._frameGen[lane].myFrameGen,
                    ))

        # Create the stream interface
        for lane in self.laneConfig:

            # Check if PGP[lane].VC[1] = Camera Image (streaming data) is enabled
            if self.enVcMask[1]:

                # Create the unbatcher
                self.unbatchers[lane] = rogue.protocols.batcher.SplitterV1()

                # Debug slave
                if dataDebug:
                    # Connect the streams
                    self.dmaStreams[lane][1] >> self.unbatchers[lane] >> self._dbg[lane]

        # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
        if self.enVcMask[0]:
            self.add(pr.LocalVariable(
                name        = 'RunState',
                description = 'Run state status, which is controlled by the StopRun() and StartRun() commands',
                mode        = 'RO',
                value       = False,
            ))

        @self.command(description  = 'Stops the triggers and blows off data in the pipeline')
        def StopRun():
            # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
            if self.enVcMask[0]:
                print ('ClinkDev.StopRun() executed')

                # Get devices
                eventBuilder = self.find(typ=batcher.AxiStreamBatcherEventBuilder)
                trigger      = self.find(typ=l2si.TriggerEventBuffer)

                # Turn off the triggering
                for devPtr in trigger:
                    devPtr.MasterEnable.set(False)

                # Flush the downstream data/trigger pipelines
                for devPtr in eventBuilder:
                    devPtr.Blowoff.set(True)

                # Update the run state status variable
                self.RunState.set(False)

        @self.command(description  = 'starts the triggers and allow steams to flow to DMA engine')
        def StartRun():
            # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
            if self.enVcMask[0]:
                print ('ClinkDev.StartRun() executed')

                # Get devices
                eventBuilder = self.find(typ=batcher.AxiStreamBatcherEventBuilder)
                trigger      = self.find(typ=l2si.TriggerEventBuffer)

                # Reset all counters
                self.CountReset()

                # Arm for data/trigger stream
                for devPtr in eventBuilder:
                    devPtr.Blowoff.set(False)
                    devPtr.SoftRst()

                # Turn on the triggering
                for devPtr in trigger:
                    devPtr.MasterEnable.set(True)

                # Update the run state status variable
                self.RunState.set(True)

    def start(self, **kwargs):
        super().start(**kwargs)

#        # Hide all the "enable" variables
#        for enableList in self.find(typ=pr.EnableVariable):
#            # Hide by default
#            enableList.hidden = True

        # Check if simulation
        if (self.dev == 'sim'):
            pass

        # Check if PGP[lane].VC[0] = SRPv3 (register access) is enabled
        elif self.enVcMask[0]:
            self.ReadAll()
            self.ReadAll()
            if (self.enableDump):
                # Dump the state of the hardware before configuration
                print(f'Dumping pre-configurations...')
                self.SaveConfig('dump/config-dump-pre-config.yml')
                self.SaveState('dump/state-dump-pre-config.yml')
                self.remoteVariableDump('dump/regdump-pre-config.txt', ['RW','WO'], False );
                # Dump the address map
                self.saveAddressMap( "dump/addressMapDump.dump" )
                self.saveAddressMap( "dump/addressMapDump.h", headerEn=True )

            # Check for PCIe FW version
            fwVersion = self.ClinkPcie.AxiPcieCore.AxiVersion.FpgaVersion.get()
            if (fwVersion != self.PcieVersionLock):
                errMsg = f"""
                    PCIe.AxiVersion.FpgaVersion = {fwVersion:#04x} != {self.PcieVersionLock:#04x}
                    Please update PCIe firmware using software/scripts/updatePcieFpga.py
                    https://github.com/slaclab/lcls2-pgp-pcie-apps/blob/master/firmware/targets/shared_config.mk
                    """
                click.secho(errMsg, bg='red')
               #raise ValueError(errMsg)

            # Check for FEB FW version
            for lane in self.laneConfig:
                # Unhide the because dependent on PGP link status
                self.ClinkFeb[lane].enable.hidden = False
                # Check for PGP link up
                if (self.RemRxLinkReady[lane].get() != 0):
                    # Check for FW version
                    fwVersion = self.ClinkFeb[lane].AxiVersion.FpgaVersion.get()
                    if (fwVersion != self.FebVersionLock):
                        errMsg = f"""
                            Fpga[lane={lane}].AxiVersion.FpgaVersion = {fwVersion:#04x} != {self.FebVersionLock:#04x}
                            Please update Fpga[{lane}] at Lane={lane} firmware using software/scripts/updateFeb.py
                            https://github.com/slaclab/cameralink-gateway/blob/master/firmware/targets/shared_version.mk
                            """
                        click.secho(errMsg, bg='red')
                        raise ValueError(errMsg)

                    # Force expand parts of the device tree
                    self.ClinkPcie.Hsio.PgpRxVcMon[lane]._expand = True
                    self.ClinkPcie.Hsio.PgpRxVcMon[lane].Ch[0]._expand = False
                    self.ClinkPcie.Hsio.PgpRxVcMon[lane].Ch[1]._expand = True # Camera image stream
                    self.ClinkPcie.Hsio.PgpRxVcMon[lane].Ch[2]._expand = False
                    self.ClinkPcie.Hsio.PgpRxVcMon[lane].Ch[3]._expand = False
                else:
                    # Collapse the lanes with links that are down
                    self.ClinkPcie.Application.AppLane[lane]._expand = False

            # Check if PGP[lane].VC[2] = Camera UART (streaming data) is enabled
            if self.enVcMask[2]:

                # Startup procedures for OPAL1000
                uartDev = self.find(typ=cl.UartOpal1000)
                for dev in uartDev:
                    pass

                # Startup procedures for Piranha4
                uartDev = self.find(typ=cl.UartPiranha4)
                for dev in uartDev:
                    dev.SendEscape()
                    dev.SPF.setDisp('0')
                    dev.GCP()

                # Startup procedures for Up900cl12b
                uartDev = self.find(typ=cl.UartUp900cl12b)
                for dev in uartDev:
                    clCh = self.find(typ=cl.ClinkChannel)
                    for clChDev in clCh:
                        clChDev.SerThrottle.set(30000)
                    dev.AM()
                    dev.SM.set('f')
                    dev.RP()

                # Startup procedures for ImperxC1921
                uartDev = self.find(typ=cl.UartImperxC1921)
                for dev in uartDev:
                    clCh = self.find(typ=cl.ClinkChannel)
                    for clChDev in clCh:
                        clChDev.BaudRate.set(115200)
                        clChDev.SerThrottle.set(10000)
                    dev.Trg_Mode_En.set('1') # 0x1 - trigger is enabled; camera in trigger mode
                    dev.Trg_Inp_Sel.set('1') # 0x3 - computer; camera expects trigger from CC1 via Camera Link cable.

                # Startup procedures for JaiCm140
                uartDev = self.find(typ=cl.UartJaiCm140)
                for dev in uartDev:
                    dev.VN.set('?') # Firmware Version
                    dev.ID.set('?') # Camera ID Request
                    dev.MD.set('?') # Model Name Request

            # Load the configurations
            if self.enableConfig:

                # Useful pointer
                timingRx = self.ClinkPcie.Hsio.TimingRx

                # Start up the timing system = LCLS-II mode
                if self.startupMode:

                    # Set the default to  LCLS-II mode
                    defaultFile = ['config/defaults_LCLS-II.yml']

                    # Startup in LCLS-II mode
                    if self.standAloneMode:
                        timingRx.ConfigureXpmMini()
                    else:
                        timingRx.ConfigLclsTimingV2()

                # Else LCLS-I mode
                else:

                    # Set the default to  LCLS-I mode
                    defaultFile = ['config/defaults_LCLS-I.yml']

                    # Startup in LCLS-I mode
                    if self.standAloneMode:
                        timingRx.ConfigureTpgMiniStream()
                    else:
                        timingRx.ConfigLclsTimingV1()

                # Read all the variables
                self.ReadAll()
                self.ReadAll()

                # Load the YAML configurations
                defaultFile.extend(self.defaultFile)
                print(f'Loading {defaultFile} Configuration File...')
                self.LoadConfig(defaultFile)

            if (self.enableDump):
                # Dump the state of the hardware before configuration
                self.ReadAll()
                self.ReadAll()
                print(f'Dumping post-configurations...')
                self.SaveConfig('dump/config-dump-post-config.yml')
                self.SaveState('dump/state-dump-post-config.yml')
                self.remoteVariableDump('dump/regdump-post-config.txt', ['RW','WO'], False );

    # Function calls after loading YAML configuration
    def initialize(self):
        super().initialize()
        self.StopRun()
        self.CountReset()
