# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'procStat.ui'
#
# Created: Fri Oct 30 15:15:46 2015
#      by: PyQt4 UI code generator 4.7.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(370, 1100)
        self.centralwidget = QtGui.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vboxlayout = QtGui.QVBoxLayout(self.centralwidget)
        self.vboxlayout.setSpacing(5)
        self.vboxlayout.setContentsMargins(0, 5, 5, 0)
        self.vboxlayout.setObjectName("vboxlayout")
        self.groupBoxProcessStatus = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.groupBoxProcessStatus.sizePolicy().hasHeightForWidth())
        self.groupBoxProcessStatus.setSizePolicy(sizePolicy)
        self.groupBoxProcessStatus.setAlignment(QtCore.Qt.AlignLeading)
        self.groupBoxProcessStatus.setFlat(True)
        self.groupBoxProcessStatus.setCheckable(False)
        self.groupBoxProcessStatus.setObjectName("groupBoxProcessStatus")
        self.vboxlayout1 = QtGui.QVBoxLayout(self.groupBoxProcessStatus)
        self.vboxlayout1.setSpacing(0)
        self.vboxlayout1.setMargin(0)
        self.vboxlayout1.setObjectName("vboxlayout1")
        self.tableProcStat = QtGui.QTableWidget(self.groupBoxProcessStatus)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableProcStat.sizePolicy().hasHeightForWidth())
        self.tableProcStat.setSizePolicy(sizePolicy)
        self.tableProcStat.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableProcStat.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.tableProcStat.setAlternatingRowColors(True)
        self.tableProcStat.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.tableProcStat.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.tableProcStat.setShowGrid(False)
        self.tableProcStat.setColumnCount(0)
        self.tableProcStat.setObjectName("tableProcStat")
        self.tableProcStat.setColumnCount(0)
        self.tableProcStat.setRowCount(0)
        self.vboxlayout1.addWidget(self.tableProcStat)
        self.horizontalFrame = QtGui.QFrame(self.groupBoxProcessStatus)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtGui.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonConsole = QtGui.QPushButton(self.horizontalFrame)
        self.pushButtonConsole.setCheckable(True)
        self.pushButtonConsole.setObjectName("pushButtonConsole")
        self.horizontalLayout.addWidget(self.pushButtonConsole)
        self.pushButtonLogfile = QtGui.QPushButton(self.horizontalFrame)
        self.pushButtonLogfile.setCheckable(True)
        self.pushButtonLogfile.setObjectName("pushButtonLogfile")
        self.horizontalLayout.addWidget(self.pushButtonLogfile)
        self.pushButtonRestart = QtGui.QPushButton(self.horizontalFrame)
        self.pushButtonRestart.setCheckable(True)
        self.pushButtonRestart.setObjectName("pushButtonRestart")
        self.horizontalLayout.addWidget(self.pushButtonRestart)
        self.vboxlayout1.addWidget(self.horizontalFrame)
        self.vboxlayout.addWidget(self.groupBoxProcessStatus)
        self.groupBoxOutputFileStatus = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.groupBoxOutputFileStatus.sizePolicy().hasHeightForWidth())
        self.groupBoxOutputFileStatus.setSizePolicy(sizePolicy)
        self.groupBoxOutputFileStatus.setFlat(True)
        self.groupBoxOutputFileStatus.setObjectName("groupBoxOutputFileStatus")
        self.vboxlayout2 = QtGui.QVBoxLayout(self.groupBoxOutputFileStatus)
        self.vboxlayout2.setSpacing(0)
        self.vboxlayout2.setMargin(0)
        self.vboxlayout2.setObjectName("vboxlayout2")
        self.textBrowser = QtGui.QTextBrowser(self.groupBoxOutputFileStatus)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setObjectName("textBrowser")
        self.vboxlayout2.addWidget(self.textBrowser)
        self.vboxlayout.addWidget(self.groupBoxOutputFileStatus)
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QtGui.QApplication.translate("mainWindow", "ProcStat", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBoxProcessStatus.setTitle(QtGui.QApplication.translate("mainWindow", "Process Status", None, QtGui.QApplication.UnicodeUTF8))
        self.tableProcStat.setSortingEnabled(False)
        self.pushButtonConsole.setToolTip(QtGui.QApplication.translate("mainWindow", "Select this button, and click on processes to open consoles", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButtonConsole.setText(QtGui.QApplication.translate("mainWindow", "Open Console", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButtonLogfile.setToolTip(QtGui.QApplication.translate("mainWindow", "Select this button, and click on processes to view logfiles", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButtonLogfile.setText(QtGui.QApplication.translate("mainWindow", "View Logfile", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButtonRestart.setToolTip(QtGui.QApplication.translate("mainWindow", "Select this button, and click on remote processes to restart", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButtonRestart.setText(QtGui.QApplication.translate("mainWindow", "Restart", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBoxOutputFileStatus.setTitle(QtGui.QApplication.translate("mainWindow", "Output File Status", None, QtGui.QApplication.UnicodeUTF8))
