#!/bin/env python

import sys
import os
import platform
import time
import getopt
import _thread
import locale
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QVariant, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCursor, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QMessageBox, QTableWidgetItem
from ProcMgr import ProcMgr, deduce_platform
import ui_procStat
import subprocess

__version__ = "0.5"

# OutDir Full Path Example: daq-sxr-ana01: /u2/pcds/pds/sxr/e19/e19-r0026-s00-c00.xtc
sOutDirPrefix1 = "/u2/pcds/pds/"
sOutFileExtension = ""
bProcMgrThreadError = False
sErrorOutDirNotExist = "Output Dir Not Exist"

localIdList = []

class CustomIOError(Exception):
  pass

def getOutputFileName(iExperiment, sExpType,eventNodes):
  if iExperiment < 0:
    return []
  if eventNodes is None:
    return []

  sExpSubDir = "e%d" % (iExperiment)
  formFileStatusDatabase = []
  zeroFilesFlag = False

  noOfEventNodes = len(eventNodes)


  for sIndex in range(noOfEventNodes):

    # form filepath, command, and ssh to remote event nodes
    sExpDir = os.path.join( sOutDirPrefix1,sExpType, sExpSubDir )
    sshCommand = '/usr/bin/ssh %(host)s ls -rtlB   %(dir)s | /usr/bin/tail -1' %   {'host': eventNodes[sIndex], 'dir': sExpDir}
    process = subprocess.Popen(sshCommand, shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output,stderr = process.communicate()
    status = process.poll()
    zeroFilesFlag = False
    if b"No such file or directory" in output:
      zeroFilesFlag = True
    elif b"total 0\n" in output:
      zeroFilesFlag = True
    elif len(output.split()) < 5:
      print('getOutputFileName: sshCommand (%s)' % sshCommand)
      print('getOutputFileName: output too short (%s)' % output)
      zeroFilesFlag = True

    # extract the output and arrange it for graphical interface
    if not zeroFilesFlag:
      splittedOutput = output.split()
      try:
        extractFileName = splittedOutput[-1].decode()
      except:
        print('ERROR:: %s' % output)
        raise CustomIOError
      completeFilePath = '%(host)s: %(dirPath)s/%(file)s '% {'host':eventNodes[sIndex], 'dirPath':sExpDir,'file':extractFileName}
      extractTime = splittedOutput[-4]+b'  '+splittedOutput[-3]+b'  '+splittedOutput[-2]
      extractSize = splittedOutput[-5]
      formFileStatusDatabase.append( { "fn": completeFilePath, "size": extractSize, "mtime": extractTime } )

  return formFileStatusDatabase

def printStackTrace():
  print( "---- Printing program call stacks for debug ----" )
  traceback.print_exc(file=sys.stdout)
  print( "------------------------------------------------" )
  return

def procMgrThreadWrapper(win, sConfigFile, iExperiment, sExpType, iPlatform, fQueryInterval, evgProcMgr,eventNodes ):
  try:
    procMgrThread(win, sConfigFile, iExperiment, sExpType, iPlatform, fQueryInterval, evgProcMgr,eventNodes )
  except:
    sErrorReport = "procMgrThreadWrapper(): procMgrThread(ConfigFile = %s, Exp Id = %d, Exp Type = %s, Platform = %d, Query Interval = %f ) Failed" %\
     (sConfigFile, iExperiment, sExpType, iPlatform, float(fQueryInterval) )
    print( sErrorReport )
    printStackTrace()
    bProcMgrThreadError = True
    win.UnknownError.emit(sErrorReport) # Send out the signal to notify the main window
  return


def procMgrThread(win, sConfigFile, iExperiment, sExpType, iPlatform, fQueryInterval, evgProcMgr,eventNodes ):

  locale.setlocale( locale.LC_ALL, "" ) # set locale for printing formatted numbers later

  while True:
    # refresh ProcMgr status
    global bProcMgrThreadError

    try:
      procMgr = ProcMgr(sConfigFile, iPlatform)
      win.procMgr = procMgr
      ldProcStatus = procMgr.getStatus()

    except IOError:
      print( "procMgrThread(): ProcMgr(%s, %d): I/O error" % (sConfigFile, iPlatform) )
      printStackTrace()
      # Send out the signal to notify the main window
      win.ProcMgrIOError.emit(sConfigFile, iPlatform )

    except:
      print( "procMgrThread(): ProcMgr(%s, %d) Failed" % (sConfigFile, iPlatform) )
      printStackTrace()
      # Send out the signal to notify the main window
      win.ProcMgrGeneralError.emit(sConfigFile, iPlatform )

    fileStatusDatabase = None  # set default value
    try:
      fileStatusDatabase = getOutputFileName( iExperiment, sExpType,eventNodes )

    except CustomIOError:
      print( "Error in ssh or Output Directory::(%s)" % (sOutDirPrefix1) )
      printStackTrace()
      win.OutputDirError.emit(sOutDirPrefix1 )    # Send out the signal to notify the main window
      
    except:
      sErrorReport = "procMgrThread(): getOutputFileName() failed due to a general error, possibly caused by filesystem disconnection\n"
      print( sErrorReport )
      printStackTrace()
      # Send out the signal to notify the main window
      win.ThreadGeneralError.emit(sErrorReport)

    if fileStatusDatabase != None:
      # Send out the signal to notify the main window
      win.Updated.emit(ldProcStatus, fileStatusDatabase, iExperiment, sExpType)
    else:
      print( "No file status available -- skipping update" )
    time.sleep(fQueryInterval )

  return

class WinProcStat(QtWidgets.QMainWindow, ui_procStat.Ui_mainWindow):

  # define 'Updated' signal
  Updated = pyqtSignal(list, list, int, str)

  # define 'UnknownError' signal
  UnknownError = pyqtSignal(str)

  # define 'ProcMgrIOError' signal
  ProcMgrIOError = pyqtSignal(str, int)

  # define 'ProcMgrGeneralError' signal
  ProcMgrGeneralError = pyqtSignal(str, int)

  # define 'ThreadGeneralError' signal
  ThreadGeneralError = pyqtSignal(str, int)

  # define 'OutputDirError' signal
  OutputDirError = pyqtSignal(str)

  def __init__(self, evgProcMgr, parent = None):
    super(WinProcStat, self).__init__(parent)

    self.sCurKey = ""
    self.setupUi(self)

    # setup message box for displaying warning messages
    self.msgBox = QMessageBox( QMessageBox.Warning, "Warning",
      "", QMessageBox.Ok, self )
    self.msgBox.setWindowModality(Qt.NonModal)

    # adjust GUI settings
    self.bFirstSort = False;

    # setup signal handlers

    # built-in signals
    self.pushButtonConsole.clicked.connect(self.onClickConsole)
    self.pushButtonLogfile.clicked.connect(self.onClickLogfile)
    self.pushButtonRestart.clicked.connect(self.onClickRestart)
    self.tableProcStat.cellClicked.connect(self.onProcCellClicked)

    # new signals defined using pyqtSignal()
    self.Updated.connect(self.onProcMgrUpdated)
    self.ProcMgrIOError.connect(self.onProcMgrIOError )
    self.ThreadGeneralError.connect(self.onThreadGeneralError )
    self.ProcMgrGeneralError.connect(self.onProcMgrGeneralError )
    self.OutputDirError.connect(self.onProcMgrOutputDirError )
    self.UnknownError.connect(self.onProcMgrUnknownError)

    self.iShowConsole = 0 # 0: Don't show console, 1: Show console, 2: Show logfile, 3: Restart
    self.procMgr = None
    return


  def closeEvent(self, event):
    self.msgBox.close()
    return

  def onProcMgrUpdated(self, ldProcStatus, ldOutputFileStatus, iExperiment, sExpType):
    self.statusbar.showMessage( "Refreshing ProcMgr status..." )

    self.tableProcStat.clear()
    self.tableProcStat.setSortingEnabled( False )
    self.tableProcStat.setRowCount(len(ldProcStatus))
    self.tableProcStat.setColumnCount(2)
    self.tableProcStat.setHorizontalHeaderLabels(["ID", "Status"])

    itemCur = None

    for iRow, dProcStatus in enumerate( ldProcStatus ):

      # col 1 : UniqueID
      if isinstance(dProcStatus["showId"], str):
        # str
        showId = dProcStatus["showId"]
      else:
        # bytes
        showId = dProcStatus["showId"].decode()
      item = QTableWidgetItem(showId)
      item.setData(Qt.UserRole, QVariant(showId))
      self.tableProcStat.setItem(iRow, 0, item)

      if showId == self.sCurKey:
        itemCur = item

      # col 2 : Status
      sStatus = dProcStatus["status"]
      item = QTableWidgetItem()
      brush1 = QBrush(QColor.fromRgb( 255, 255, 255 ) )
      item.setForeground(brush1)
      if ( sStatus == ProcMgr.STATUS_NOCONNECT ):
        item.setData(0, QVariant(" NO CONNECT"))
        item.setBackground( QBrush(QColor.fromRgb( 0, 0, 192 ) ) )
      elif ( sStatus == ProcMgr.STATUS_RUNNING ):
        item.setData(0, QVariant("RUNNING"))
        item.setBackground( QBrush(QColor.fromRgb( 0, 192, 0 ) ) )
      elif ( sStatus == ProcMgr.STATUS_SHUTDOWN ):
        item.setData(0, QVariant(" SHUTDOWN"))
        item.setBackground( QBrush(QColor.fromRgb( 192, 192, 0 ) ) )
      elif ( sStatus == ProcMgr.STATUS_ERROR ):
        item.setData(0, QVariant(" ERROR"))
        item.setBackground( QBrush(QColor.fromRgb( 255, 0, 0 ) ) )

      self.tableProcStat.setItem(iRow, 1, item)

    # end for iRow, key in enumerate( sorted(procMgr.d.iterkeys()) ):

    self.tableProcStat.setColumnWidth(0,160)
    self.tableProcStat.setColumnWidth(1,100)

    if not self.bFirstSort:
      self.bFirstSort = True
      self.tableProcStat.sortItems( 1, Qt.AscendingOrder )

    self.tableProcStat.setSortingEnabled( True )
    if itemCur != None:
      self.tableProcStat.setCurrentItem( itemCur )

    if ldOutputFileStatus:
      sOutputStatus = ""
      for dOutputFileStatus in ldOutputFileStatus:
        sOutputStatus += """
<p><b>Filename:</b
> %s <br>
<b>Size:</b> %s Bytes <br>
<b>Last Modification Time:</b> %s
""" % ( dOutputFileStatus["fn"], dOutputFileStatus["size"].decode(), dOutputFileStatus["mtime"].decode() )

      # save the scrollar positions
      hVal = self.textBrowser.horizontalScrollBar().value()
      vVal = self.textBrowser.verticalScrollBar().value()

      self.textBrowser.setHtml( sOutputStatus )

      # restore the scrollar positions
      self.textBrowser.horizontalScrollBar().setValue(hVal)
      self.textBrowser.verticalScrollBar().setValue(vVal)
    else:
      self.textBrowser.setHtml( "No output file is found for experiment type <b>%s</b> id <b>%d</b>" % (sExpType, iExperiment) )

    self.statusbar.clearMessage()
    return

  @pyqtSlot(int, int)
  def onProcCellClicked(self, iRow, iCol):
    global localIdList

    if self.procMgr == None:
      return
    if self.iShowConsole == 0:
      return

    if len(localIdList) == 0:
      for kee in self.procMgr.d.keys():
        id = kee.replace('localhost:', '')
        if id != kee:
          localIdList.append(id)

    item = self.tableProcStat.item(iRow, 0)
    showId = item.data(Qt.UserRole)
    if self.iShowConsole == 1:
      self.procMgr.spawnConsole(showId, False)
    elif self.iShowConsole == 2:
      self.procMgr.spawnLogfile(showId, False)
    elif self.iShowConsole == 3:
      if showId in localIdList:
          self.showWarningWindow("Not Supported",
                                 "Local processes cannot be individually restarted")
      else:
          # determine logpathbase and coresize
          logpathbase = None
          coresize = 0
          if 'LOGPATH' in self.procMgr.procmgr_macro:
            logpathbase = self.procMgr.procmgr_macro['LOGPATH']
          if 'CORESIZE' in self.procMgr.procmgr_macro:
            coresize = int(self.procMgr.procmgr_macro['CORESIZE'])

          # override cursor
          QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

          # stop individual process
          self.procMgr.stop([ showId ], True)
          time.sleep(0.5)

          # start individual process
          self.procMgr.start([ showId ], True, logpathbase, coresize)

          # restore cursor
          QApplication.restoreOverrideCursor()
    return

  def onClickConsole(self, bChecked):
    self.pushButtonLogfile.setChecked(False)
    self.pushButtonRestart.setChecked(False)
    if bChecked:
      self.iShowConsole = 1
    else:
      self.iShowConsole = 0

  def onClickLogfile(self, bChecked):
    self.pushButtonConsole.setChecked(False)
    self.pushButtonRestart.setChecked(False)
    if bChecked:
      self.iShowConsole = 2
    else:
      self.iShowConsole = 0

  def onClickRestart(self, bChecked):
    self.pushButtonConsole.setChecked(False)
    self.pushButtonLogfile.setChecked(False)
    if bChecked:
      self.iShowConsole = 3
    else:
      self.iShowConsole = 0

  @pyqtSlot(str, int)
  def onProcMgrIOError(self, sConfigFile, iPlatform):
    self.showWarningWindow( "ProcMgr IO Error",
     "<p>ProcMgr config file <b>%s</b> (platform <b>%d</b>) cannot be processed correctly, due to an IO Error.<p>" % (sConfigFile, iPlatform) +
     "<p>Please check the input file format, or specify a new config file instead." )
    return

  @pyqtSlot(str, int)
  def onProcMgrGeneralError(self, sConfigFile, iPlatform):
    self.showWarningWindow( "ProcMgr General Error",
     "<p>ProcMgr config file <b>%s</b> (platform <b>%d</b>) cannot be processed correctly, due to a general Error.<p>" % (sConfigFile, iPlatform) +
     "<p>Please check the input file content, and the target machine status." )
    return

  @pyqtSlot(str)
  def onProcMgrOutputDirError(self, sOutDirPrefix1):
    self.showWarningWindow( "Output Directory Does Not Exist",
     "<p>Please check if the system has access to output directory <i>%s</i>.<p>" % (sOutDirPrefix1) )
    return

  @pyqtSlot(str)
  def onThreadGeneralError(self, sErrorReport):
    self.showWarningWindow( "Thread General Error",
     "<p><i>%s</i><p>" % (sErrorReport) +
     "<p>ProcMgr thread had a general error. Please check the log file for more details.\n")
    return

  @pyqtSlot(str)
  def onProcMgrUnknownError(self, sErrorReport):
    QMessageBox.critical(self, "Unknown Error",
     "<p><i>%s</i><p>" % (sErrorReport) +
     "<p>ProcStat is not updating the process status. Please check the log file for more details.\n")
    self.close()
    return

  def showWarningWindow( self, title, text ):
    self.msgBox.setWindowTitle( title )
    self.msgBox.setText( text )
    self.msgBox.show()
    return

  @pyqtSlot(int,int,int,int)
  def on_tableProcStat_currentCellChanged( self, iCurRow, iCurCol, iPrevRow, iPrevCol ):
    if iCurRow < 0: return

    itemCur = self.tableProcStat.item(iCurRow,0)
    if itemCur == None: return

    self.sCurKey = itemCur.data(Qt.UserRole)
    return

  @pyqtSlot()
  def on_actionOpen_triggered(self):
    sFnConfig = str(QFileDialog.getOpenFileName(self, \
                        "ProcMgr Config File", ".", \
                        "config files (*.cnf)" ))
    return

  @pyqtSlot()
  def on_actionQuit_triggered(self):
    self.close()
    return

  @pyqtSlot()
  def on_actionAbout_triggered(self):
    QMessageBox.about(self, "About procStat",
            """<b>ProcMgr Status Monitor</b> v %s
            <p>Copyright &copy; 2009 SLAC PCDS
            <p>This application is used to monitor procMgr status and report output file status.
            <p>Python %s - Qt %s - PyQt %s on %s""" % (
            __version__, platform.python_version(),
            QT_VERSION_STR, PYQT_VERSION_STR, platform.system()))
    return

def showUsage():
  print( """\
Usage: %s  [-e | --experiment <Experiment Id>]  [-t | --type <Experiment Type>]  [-p | --platform <Platform Id>]  [-i | --interval <ProcMgr Query Interval>]  <Config file>
  -e | --experiment <Experiment Id>*               Set experiment id (default: No id, and no output file status displayed)
  -t | --type       <Experiment Type>*             Set experiment type (default: amo)
  -p | --platform   <Platform Id>                  Set platform id (default: id is deduced from config file)
  -i | --interval   <ProcMgr Query Interval>       Query interval in seconds (default: 3 seconds)
  -n | --eventnodes <eventNode0+eventNode1+...>*   Names of Event Nodes conected with '+' sign

Program Version %s\
""" % ( __file__, __version__ ) )
  return

def main():
  iExperiment = 0
  sExpType = "amo"
  iPlatform = -1
  fProcmgrQueryInterval = 5.0
  eventNodes = []
  eventNodesDefined = False
  exptTypeDefined = False
  exptIdDefined = False

  (llsOptions, lsRemainder) = getopt.getopt(sys.argv[1:], \
   "e:t:vhp:i:n:", \
   ["experiment", "type", "version", "help", "platform=", "interval=","eventnodes"])

  for (sOpt, sArg) in llsOptions:
    if sOpt in ("-e", "--experiment" ):
      iExperiment = int(sArg)
      exptIdDefined = True
    elif sOpt in ("-t", "--type" ):
      sExpType = sArg
      exptTypeDefined = True
    elif sOpt in ("-n", "--eventnodes" ):
      eventNodes = sArg.split('+')
      print('eventnodes = %s' % eventNodes)
      eventNodesDefined = True
    elif sOpt in ("-v", "-h", "--version", "--help" ):
      showUsage()
      return 1
    elif sOpt in ('-p', '--platform' ):
      iPlatform = int(sArg)
    elif sOpt in ('-i', '--interval' ):
      fProcmgrQueryInterval = float(sArg)


  if not exptTypeDefined:
    print('Expt Type Not Defined -- using default value \''+sExpType+'\'')
  if not exptIdDefined:
    print('Expt ID Not Defined -- using default value \'%d\'' % iExperiment)


  if len(lsRemainder) < 1:
    print( __file__ + ": Config file is not specified" )
    showUsage()
    return 1

  sConfigFile = lsRemainder[0]

  if (iPlatform < 0):
    try:
        iPlatform = deduce_platform(sConfigFile)
    except IOError:
        raise CustomIOError("main(): I/O error while reading " +  sConfigFile)

  evgProcMgr = QObject()

  app = QApplication([])
  app.setOrganizationName("SLAC")
  app.setOrganizationDomain("slac.stanford.edu")
  app.setApplicationName("procStat")
  win = WinProcStat( evgProcMgr )
  win.show()

  _thread.start_new_thread( procMgrThreadWrapper, (win, sConfigFile, iExperiment, sExpType, iPlatform, fProcmgrQueryInterval, evgProcMgr,eventNodes) )

  app.exec_()

  return

# Main Entry
if __name__ == "__main__":
  iRet = 0

  try:
    iRet = main()
  except:
    iRet = 101
    print(__file__ + ": %s" % (sys.exc_info()[1]))
    print( "---- Printing program call stacks for debug ----" )
    traceback.print_exc(file=sys.stdout)
    print( "------------------------------------------------" )
    showUsage()

  sys.exit(iRet)
