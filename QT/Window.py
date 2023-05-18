# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1111, 744)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.main_tab = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.main_tab.setFont(font)
        self.main_tab.setObjectName("main_tab")
        self.db_tab = QtWidgets.QWidget()
        self.db_tab.setObjectName("db_tab")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.db_tab)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.connect_table = QtWidgets.QTableWidget(self.db_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.connect_table.sizePolicy().hasHeightForWidth())
        self.connect_table.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.connect_table.setFont(font)
        self.connect_table.setColumnCount(2)
        self.connect_table.setObjectName("connect_table")
        self.connect_table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.connect_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignVCenter)
        self.connect_table.setHorizontalHeaderItem(1, item)
        self.verticalLayout_6.addWidget(self.connect_table)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.load_Button = QtWidgets.QPushButton(self.db_tab)
        self.load_Button.setObjectName("load_Button")
        self.horizontalLayout.addWidget(self.load_Button)
        spacerItem = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.delete_Button = QtWidgets.QPushButton(self.db_tab)
        self.delete_Button.setObjectName("delete_Button")
        self.horizontalLayout.addWidget(self.delete_Button)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.horizontalLayout_8.addLayout(self.verticalLayout_6)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.user_group = QtWidgets.QGroupBox(self.db_tab)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.user_group.setFont(font)
        self.user_group.setObjectName("user_group")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.user_group)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem1, 1, 1, 1, 1)
        self.user_Edit = QtWidgets.QLineEdit(self.user_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.user_Edit.setFont(font)
        self.user_Edit.setObjectName("user_Edit")
        self.gridLayout_2.addWidget(self.user_Edit, 2, 2, 1, 1)
        self.passwd_label = QtWidgets.QLabel(self.user_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.passwd_label.setFont(font)
        self.passwd_label.setObjectName("passwd_label")
        self.gridLayout_2.addWidget(self.passwd_label, 4, 0, 1, 2)
        self.passwd_Edit = QtWidgets.QLineEdit(self.user_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.passwd_Edit.setFont(font)
        self.passwd_Edit.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.passwd_Edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.passwd_Edit.setObjectName("passwd_Edit")
        self.gridLayout_2.addWidget(self.passwd_Edit, 4, 2, 1, 1)
        self.pdsave_Check = QtWidgets.QCheckBox(self.user_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.pdsave_Check.setFont(font)
        self.pdsave_Check.setObjectName("pdsave_Check")
        self.gridLayout_2.addWidget(self.pdsave_Check, 6, 0, 1, 3)
        self.user_label = QtWidgets.QLabel(self.user_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.user_label.setFont(font)
        self.user_label.setObjectName("user_label")
        self.gridLayout_2.addWidget(self.user_label, 2, 0, 1, 2)
        self.name_label = QtWidgets.QLabel(self.user_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.name_label.setFont(font)
        self.name_label.setObjectName("name_label")
        self.gridLayout_2.addWidget(self.name_label, 0, 0, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem2, 5, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem3, 3, 1, 1, 1)
        self.name_Edit = QtWidgets.QLineEdit(self.user_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.name_Edit.setFont(font)
        self.name_Edit.setObjectName("name_Edit")
        self.gridLayout_2.addWidget(self.name_Edit, 0, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.user_group)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem4)
        self.line = QtWidgets.QFrame(self.db_tab)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        spacerItem5 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem5)
        self.server_group = QtWidgets.QGroupBox(self.db_tab)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.server_group.setFont(font)
        self.server_group.setObjectName("server_group")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.server_group)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem6 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem6, 1, 0, 1, 1)
        self.port_Edit = QtWidgets.QLineEdit(self.server_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.port_Edit.setFont(font)
        self.port_Edit.setObjectName("port_Edit")
        self.gridLayout_3.addWidget(self.port_Edit, 2, 1, 1, 1)
        self.sid_Edit = QtWidgets.QLineEdit(self.server_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.sid_Edit.setFont(font)
        self.sid_Edit.setObjectName("sid_Edit")
        self.gridLayout_3.addWidget(self.sid_Edit, 4, 1, 1, 1)
        self.host_Edit = QtWidgets.QLineEdit(self.server_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.host_Edit.setFont(font)
        self.host_Edit.setObjectName("host_Edit")
        self.gridLayout_3.addWidget(self.host_Edit, 0, 1, 1, 1)
        self.port_label = QtWidgets.QLabel(self.server_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.port_label.setFont(font)
        self.port_label.setObjectName("port_label")
        self.gridLayout_3.addWidget(self.port_label, 2, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem7, 3, 0, 1, 1)
        self.host_label = QtWidgets.QLabel(self.server_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.host_label.setFont(font)
        self.host_label.setObjectName("host_label")
        self.gridLayout_3.addWidget(self.host_label, 0, 0, 1, 1)
        self.sid_label = QtWidgets.QLabel(self.server_group)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.sid_label.setFont(font)
        self.sid_label.setObjectName("sid_label")
        self.gridLayout_3.addWidget(self.sid_label, 4, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.server_group)
        spacerItem8 = QtWidgets.QSpacerItem(20, 200, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem8)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        self.connect_Button = QtWidgets.QPushButton(self.db_tab)
        self.connect_Button.setObjectName("connect_Button")
        self.horizontalLayout_2.addWidget(self.connect_Button)
        self.save_Button = QtWidgets.QPushButton(self.db_tab)
        self.save_Button.setObjectName("save_Button")
        self.horizontalLayout_2.addWidget(self.save_Button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8.addLayout(self.verticalLayout_2)
        self.main_tab.addTab(self.db_tab, "")
        self.detect_tab = QtWidgets.QWidget()
        self.detect_tab.setObjectName("detect_tab")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.detect_tab)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.current_group = QtWidgets.QGroupBox(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_group.sizePolicy().hasHeightForWidth())
        self.current_group.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.current_group.setFont(font)
        self.current_group.setObjectName("current_group")
        self.gridLayout = QtWidgets.QGridLayout(self.current_group)
        self.gridLayout.setObjectName("gridLayout")
        self.current = QtWidgets.QWidget(self.current_group)
        self.current.setObjectName("current")
        self.gridLayout.addWidget(self.current, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.current_group)
        self.voltage_group = QtWidgets.QGroupBox(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.voltage_group.sizePolicy().hasHeightForWidth())
        self.voltage_group.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.voltage_group.setFont(font)
        self.voltage_group.setObjectName("voltage_group")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.voltage_group)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.voltage = QtWidgets.QWidget(self.voltage_group)
        self.voltage.setObjectName("voltage")
        self.gridLayout_4.addWidget(self.voltage, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.voltage_group)
        self.horizontalLayout_7.addLayout(self.verticalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.mtf_group = QtWidgets.QGroupBox(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.mtf_group.sizePolicy().hasHeightForWidth())
        self.mtf_group.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.mtf_group.setFont(font)
        self.mtf_group.setObjectName("mtf_group")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.mtf_group)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.mtf = QtWidgets.QWidget(self.mtf_group)
        self.mtf.setObjectName("mtf")
        self.gridLayout_5.addWidget(self.mtf, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.mtf_group)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.cycle_label = QtWidgets.QLabel(self.detect_tab)
        self.cycle_label.setObjectName("cycle_label")
        self.horizontalLayout_4.addWidget(self.cycle_label)
        self.cycle_val = QtWidgets.QLabel(self.detect_tab)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.cycle_val.setFont(font)
        self.cycle_val.setText("")
        self.cycle_val.setObjectName("cycle_val")
        self.horizontalLayout_4.addWidget(self.cycle_val)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem10)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.num_Slider = QtWidgets.QSlider(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.num_Slider.sizePolicy().hasHeightForWidth())
        self.num_Slider.setSizePolicy(sizePolicy)
        self.num_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.num_Slider.setObjectName("num_Slider")
        self.horizontalLayout_6.addWidget(self.num_Slider)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.num_Edit = QtWidgets.QLineEdit(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.num_Edit.sizePolicy().hasHeightForWidth())
        self.num_Edit.setSizePolicy(sizePolicy)
        self.num_Edit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.num_Edit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.num_Edit.setObjectName("num_Edit")
        self.horizontalLayout_5.addWidget(self.num_Edit)
        self.max_num_label = QtWidgets.QLabel(self.detect_tab)
        self.max_num_label.setObjectName("max_num_label")
        self.horizontalLayout_5.addWidget(self.max_num_label)
        self.num_Button = QtWidgets.QPushButton(self.detect_tab)
        self.num_Button.setObjectName("num_Button")
        self.horizontalLayout_5.addWidget(self.num_Button)
        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.groupBox = QtWidgets.QGroupBox(self.detect_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(20)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 140))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.fault_type_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fault_type_label.sizePolicy().hasHeightForWidth())
        self.fault_type_label.setSizePolicy(sizePolicy)
        self.fault_type_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.fault_type_label.setObjectName("fault_type_label")
        self.gridLayout_6.addWidget(self.fault_type_label, 0, 0, 1, 1)
        self.fault_type = QtWidgets.QLabel(self.groupBox)
        self.fault_type.setText("")
        self.fault_type.setObjectName("fault_type")
        self.gridLayout_6.addWidget(self.fault_type, 0, 1, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.detect_Button = QtWidgets.QPushButton(self.groupBox)
        self.detect_Button.setObjectName("detect_Button")
        self.verticalLayout_5.addWidget(self.detect_Button)
        self.detectall_Button = QtWidgets.QPushButton(self.groupBox)
        self.detectall_Button.setObjectName("detectall_Button")
        self.verticalLayout_5.addWidget(self.detectall_Button)
        self.stop_Button = QtWidgets.QPushButton(self.groupBox)
        self.stop_Button.setObjectName("stop_Button")
        self.verticalLayout_5.addWidget(self.stop_Button)
        self.gridLayout_6.addLayout(self.verticalLayout_5, 0, 3, 2, 1)
        self.acc_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.acc_label.sizePolicy().hasHeightForWidth())
        self.acc_label.setSizePolicy(sizePolicy)
        self.acc_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.acc_label.setObjectName("acc_label")
        self.gridLayout_6.addWidget(self.acc_label, 1, 0, 1, 1)
        self.acc = QtWidgets.QLabel(self.groupBox)
        self.acc.setText("")
        self.acc.setObjectName("acc")
        self.gridLayout_6.addWidget(self.acc, 1, 1, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem11, 1, 2, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout_7.addLayout(self.verticalLayout)
        self.main_tab.addTab(self.detect_tab, "")
        self.horizontalLayout_3.addWidget(self.main_tab)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.main_tab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.name_Edit, self.user_Edit)
        MainWindow.setTabOrder(self.user_Edit, self.passwd_Edit)
        MainWindow.setTabOrder(self.passwd_Edit, self.pdsave_Check)
        MainWindow.setTabOrder(self.pdsave_Check, self.host_Edit)
        MainWindow.setTabOrder(self.host_Edit, self.port_Edit)
        MainWindow.setTabOrder(self.port_Edit, self.sid_Edit)
        MainWindow.setTabOrder(self.sid_Edit, self.main_tab)
        MainWindow.setTabOrder(self.main_tab, self.connect_table)
        MainWindow.setTabOrder(self.connect_table, self.connect_Button)
        MainWindow.setTabOrder(self.connect_Button, self.save_Button)
        MainWindow.setTabOrder(self.save_Button, self.num_Slider)
        MainWindow.setTabOrder(self.num_Slider, self.num_Edit)
        MainWindow.setTabOrder(self.num_Edit, self.num_Button)
        MainWindow.setTabOrder(self.num_Button, self.detect_Button)
        MainWindow.setTabOrder(self.detect_Button, self.detectall_Button)
        MainWindow.setTabOrder(self.detectall_Button, self.stop_Button)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "卫星电源系统太阳阵故障检测系统"))
        item = self.connect_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "连接名"))
        item = self.connect_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "连接详细资料"))
        self.load_Button.setText(_translate("MainWindow", "加载"))
        self.delete_Button.setText(_translate("MainWindow", "删除"))
        self.user_group.setTitle(_translate("MainWindow", "用户信息"))
        self.passwd_label.setText(_translate("MainWindow", "口令："))
        self.pdsave_Check.setText(_translate("MainWindow", "保存口令"))
        self.user_label.setText(_translate("MainWindow", "用户名："))
        self.name_label.setText(_translate("MainWindow", "连接名："))
        self.server_group.setTitle(_translate("MainWindow", "服务端信息"))
        self.port_label.setText(_translate("MainWindow", "端口号："))
        self.host_label.setText(_translate("MainWindow", "主机名："))
        self.sid_label.setText(_translate("MainWindow", "SID："))
        self.connect_Button.setText(_translate("MainWindow", "连接"))
        self.save_Button.setText(_translate("MainWindow", "保存"))
        self.main_tab.setTabText(self.main_tab.indexOf(self.db_tab), _translate("MainWindow", "Oracle数据库连接"))
        self.current_group.setTitle(_translate("MainWindow", "太阳阵输入电流"))
        self.voltage_group.setTitle(_translate("MainWindow", "分流域电压"))
        self.mtf_group.setTitle(_translate("MainWindow", "马尔可夫变迁场(MTF)数据升维"))
        self.cycle_label.setText(_translate("MainWindow", "周期："))
        self.max_num_label.setText(_translate("MainWindow", "/700"))
        self.num_Button.setText(_translate("MainWindow", "确认"))
        self.groupBox.setTitle(_translate("MainWindow", "故障检测"))
        self.fault_type_label.setText(_translate("MainWindow", "类别："))
        self.detect_Button.setText(_translate("MainWindow", "检测当前样本"))
        self.detectall_Button.setText(_translate("MainWindow", "检测全部样本"))
        self.stop_Button.setText(_translate("MainWindow", "停止检测"))
        self.acc_label.setText(_translate("MainWindow", "准确率："))
        self.main_tab.setTabText(self.main_tab.indexOf(self.detect_tab), _translate("MainWindow", "太阳阵故障检测"))
