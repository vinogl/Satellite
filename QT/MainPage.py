from Window import Ui_MainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from pyts.image import MarkovTransitionField
import cx_Oracle
import json
import os
import warnings

warnings.filterwarnings('ignore')  # 抑制警告显示

# 设置参数
digital_map = {'normal': 0, 'fault': 1, 'fault_1': 2, 'fault_2': 3, 'fault_3': 4}
word_map = {'normal': '无故障', 'fault_1': '电路故障', 'fault_2': '展开机构故障', 'fault_3': '对日定向驱动机构故障'}
data_type = 'normal'

# 读取数据
accuracy = np.load('accuracy.npy')
sample_time = np.load('../numpy/sample_time.npy')
current_data = np.load('../numpy/sample_fault.npy')[digital_map[data_type]]
voltage_data = np.load('../numpy/sample_1006.npy')[1]
va_data = np.load('../numpy/V-A/%s.npy' % data_type)
acc = accuracy[digital_map[data_type]]

# 计算周期与最大样本数
cycle = 5670
max_num = int(sample_time.shape[0] / (cycle / 10))


class MainPage(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainPage, self).__init__()
        self.setupUi(self)

        # 数据库功能
        self.connect_Button.clicked.connect(self.db_connect)
        self.save_Button.clicked.connect(self.db_save)
        self.load_Button.clicked.connect(self.db_load)
        self.delete_Button.clicked.connect(self.db_delete)
        self.table_load()

        # 设置周期与最大样本数
        self.cycle_val.setText('%ds' % cycle)
        self.max_num_label.setText('/%s' % str(max_num))

        # 设置滑动条
        self.num_Slider.setMaximum(max_num)
        self.num_Slider.setMinimum(0)
        self.num_Slider.valueChanged.connect(self.slider_changed)
        self.num_Button.clicked.connect(self.num_changed)

        # 检测当前样本
        self.detect_Button.clicked.connect(self.detect_single)

        # 自动检测
        self.detectall_Button.clicked.connect(self.start_detect_all)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_all)
        self.stop_Button.clicked.connect(self.detect_stop)

        # 将画布添加到窗口中
        self.fig_c = Figure()
        self.canvas_c = FigureCanvas(self.fig_c)
        self.plot_widget_c = self.findChild(QWidget, 'current')
        self.layout_c = QVBoxLayout(self.plot_widget_c)
        self.layout_c.addWidget(self.canvas_c)

        self.fig_v = Figure()
        self.canvas_v = FigureCanvas(self.fig_v)
        self.plot_widget_v = self.findChild(QWidget, 'voltage')
        self.layout_v = QVBoxLayout(self.plot_widget_v)
        self.layout_v.addWidget(self.canvas_v)

        self.fig_m = Figure()
        self.canvas_m = FigureCanvas(self.fig_m)
        self.plot_widget_m = self.findChild(QWidget, 'mtf')
        self.layout_m = QVBoxLayout(self.plot_widget_m)
        self.layout_m.addWidget(self.canvas_m)

    def db_connect(self):
        user = self.user_Edit.text()
        password = self.passwd_Edit.text()
        host = self.host_Edit.text()
        port = self.port_Edit.text()
        sid = self.sid_Edit.text()

        try:
            connect = cx_Oracle.connect(user, password, host + ':' + port + '/' + sid)
            QMessageBox.information(self, '提示', '连接成功', QMessageBox.Yes)
        except:
            QMessageBox.information(self, '提示', '连接失败', QMessageBox.Yes)

    def db_save(self):
        connect_name = self.name_Edit.text()
        user = self.user_Edit.text()
        passwd = self.passwd_Edit.text()
        host = self.host_Edit.text()
        port = self.port_Edit.text()
        sid = self.sid_Edit.text()

        message = '%s@%s:%s/%s' % (user, host, port, sid)  # 用于显示在表格中

        # 将信息写入json文件，向下写入
        if os.path.exists('connect.json') and os.path.getsize('connect.json'):
            with open('connect.json', 'r', encoding='utf-8') as f:
                connect_info = json.load(f)
        else:
            connect_info = {}

        if self.pdsave_Check.isChecked():
            connect_info[connect_name] = {'user': user, 'password': passwd, 'host': host, 'port': port, 'sid': sid}
        else:
            connect_info[connect_name] = {'user': user, 'host': host, 'port': port, 'sid': sid}

        with open('connect.json', 'w', encoding='utf-8') as f:
            json.dump(connect_info, f, indent=4)

        self.table_load()

    def table_load(self):
        self.connect_table.setRowCount(0)
        if os.path.exists('connect.json'):
            with open('connect.json', 'r') as f:
                connect_info = json.load(f)
                for key, value in connect_info.items():
                    message = '%s@%s:%s/%s' % (value['user'], value['host'], value['port'], value['sid'])
                    self.connect_table.insertRow(0)
                    self.connect_table.setItem(0, 0, QTableWidgetItem(key))
                    self.connect_table.setItem(0, 1, QTableWidgetItem(message))

    def db_load(self):
        connect_name = self.connect_table.item(self.connect_table.currentRow(), 0).text()
        with open('connect.json', 'r', encoding='utf-8') as f:
            connect_info = json.load(f)
            self.name_Edit.setText(connect_name)
            self.user_Edit.setText(connect_info[connect_name]['user'])
            if 'password' in connect_info[connect_name].keys():
                self.passwd_Edit.setText(connect_info[connect_name]['password'])
            self.host_Edit.setText(connect_info[connect_name]['host'])
            self.port_Edit.setText(connect_info[connect_name]['port'])
            self.sid_Edit.setText(connect_info[connect_name]['sid'])

    def db_delete(self):

        # 弹窗提示是否删除
        reply = QMessageBox.question(self, '提示', '是否删除该连接信息？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            pass
        else:
            connect_name = self.connect_table.item(self.connect_table.currentRow(), 0).text()
            self.connect_table.removeRow(self.connect_table.currentRow())
            with open('connect.json', 'r', encoding='utf-8') as f:
                connect_info = json.load(f)
                connect_info.pop(connect_name)
            with open('connect.json', 'w', encoding='utf-8') as f:
                json.dump(connect_info, f)

    def plot_current(self, x, y):
        self.fig_c.clear()
        ax = self.fig_c.add_subplot(111)
        ax.set_xlabel('时间(s)',  fontdict={'family': 'SimHei', 'size': 12})
        ax.set_ylabel('太阳阵输入电流(A)',  fontdict={'family': 'SimHei', 'size': 12})
        ax.plot(x, y)
        self.fig_c.tight_layout()
        self.canvas_c.draw_idle()

    def plot_voltage(self, x, y):
        self.fig_v.clear()
        ax = self.fig_v.add_subplot(111)
        # 设置ax正常显示中文
        ax.set_xlabel('时间(s)', fontdict={'family': 'SimHei', 'size': 12})
        ax.set_ylabel('分流域电压(V)',  fontdict={'family': 'SimHei', 'size': 12})
        ax.plot(x, y)
        self.fig_v.tight_layout()
        self.canvas_v.draw_idle()

    def plot_mtf(self, data):
        self.fig_m.clear()
        function = MarkovTransitionField()
        X_tran = function.fit_transform([data])
        ax = self.fig_m.add_subplot(111)
        ax.axis('off')
        ax.imshow(X_tran[0], cmap='rainbow', origin='lower')
        self.fig_m.tight_layout()
        self.canvas_m.draw_idle()

    def plot_all(self, num):
        self.plot_current(sample_time[(num - 1) * 567:num * 567], current_data[(num - 1) * 567:num * 567])
        self.plot_voltage(sample_time[(num - 1) * 567:num * 567], voltage_data[(num - 1) * 567:num * 567])
        self.plot_mtf(va_data[(num - 1) * 567:num * 567])

    def plot_clear(self):
        self.fig_c.clear()
        self.fig_v.clear()
        self.fig_m.clear()
        self.canvas_c.draw_idle()
        self.canvas_v.draw_idle()
        self.canvas_m.draw_idle()

    def slider_changed(self):
        num = self.num_Slider.value()
        self.num_Edit.setText(str(num))

        if self.num_Slider.value() == 0:
            self.plot_clear()
        else:
            self.plot_all(num)

    def num_changed(self):
        num = int(self.num_Edit.text())
        self.num_Slider.setValue(num)

        if self.num_Slider.value() == 0:
            self.plot_clear()
        else:
            self.plot_all(num)

    def detect_sample(self):
        num = self.num_Slider.value()
        self.fault_type.setText(word_map[data_type])
        self.acc.setText("{:.2%}".format(acc[num - 1]))

    def detect_single(self):
        self.fault_type.setText('正在检测...')
        self.acc.setText('正在检测...')

        self.single_timer = QTimer()
        self.single_timer.setSingleShot(True)
        self.single_timer.timeout.connect(self.detect_sample)
        self.single_timer.start(8000)

    def start_detect_all(self):
        self.start_spot = self.num_Slider.value()
        self.timer.start(8000)

    def detect_stop(self):
        self.timer.stop()
        with open('report.txt', 'w', encoding='utf-8') as f:
            for i in range(self.start_spot, self.num_Slider.value()):
                f.write(
                    '%ds~%ds, 分类类别: %s, 准确率: %.2f%% \n' % (i * cycle, (i + 1) * cycle, word_map[data_type], acc[i] * 100))

    def detect_all(self):
        num = self.num_Slider.value()

        if num < max_num:
            num += 1
            self.num_Slider.setValue(num)
            if num == 0:
                pass
            else:
                self.detect_sample()
        else:
            self.detect_stop()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = MainPage()
    window.show()
    sys.exit(app.exec_())
