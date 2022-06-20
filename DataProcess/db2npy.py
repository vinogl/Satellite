import cx_Oracle
import datetime
import numpy as np


class ProcessModel:
    """
    数据的读取与预处理模型
    """

    def __init__(self, table, columns):
        self.table = table  # 预处理数据的表名
        self.columns = columns  # 预处理数据的列名

    def connect_database(self):
        # 连接到数据库，并生成cursor对象
        self.conn = cx_Oracle.connect('gongli', '111', '192.168.199.133:1521/ORCL')
        self.cursor = self.conn.cursor()
        self.time_column = 'CSSJ'

    def read_data(self):
        """
        读取数据，并剔除重复数据
        return:
        time_list: 数据对应的时间序列
        data_np: 选中的数据
        """
        self.cursor.execute('select %s, %s from %s order by %s' % (self.time_column, self.columns, self.table, self.time_column))

        time_dic = {}  # 时间对应数据行的映射
        data_list = []  # 各行数据

        for i, item in enumerate(self.cursor):
            dic_len_before = len(time_dic)  # 新增元素前的词典长度
            time_i = item[0]
            time_dic.update({time_i: i})
            dic_len_after = len(time_dic)  # 新增元素后的词典长度

            if dic_len_before != dic_len_after:  # 可避免重复时间点的数据录入
                row_float = [float(string) for string in item[1:]]  # 将string转为float
                data_list.append(row_float)

        time_list = list(time_dic.keys())
        data_np = np.array(data_list).transpose(1, 0)

        return time_list, data_np

    @staticmethod
    def generate_time_series(time_list, return_option='seconds_series'):
        """
        根据read_data返回的time_list，得到时间序列
        return_option='time_series': 返回距离self.zero_time的秒序列
        return_option='diff_series': 返回时间点间的间隔序列
        """
        zero_time_str = '2019-02-01 00:00:00.0'  # 设定时间的零点
        zero_time = datetime.datetime.strptime(zero_time_str, '%Y-%m-%d %H:%M:%S.0')  # 字符串转换成时间

        seconds_series = []  # 计数点秒数
        diff_series = []  # 计数间秒间隔序列

        if return_option == 'seconds_series':
            for time in time_list:
                time_sec = float((time - zero_time).total_seconds())
                seconds_series.append(time_sec)  # 将每个时刻加入时间序列
            return np.array(seconds_series)

        elif return_option == 'diff_series':
            ex_sec = zero_time
            for time in time_list:
                sec = time
                diff = float((sec - ex_sec).total_seconds())  # 计算该时刻与前一时刻的时间差
                ex_sec = sec  # 本时刻时间赋值给下一时刻的前一刻时间
                diff_series.append(diff)
            return np.array(diff_series)

        else:
            exit('return_option只可选seconds_series、diff_series')


def re_sample(second_series, data_np, sample_interval=10):
    """
    根据给定的采样间隔，对原始数据重采样。
    不同表的采样间隔不同，用此方法，可将不同表的数据统一处理。
    """
    # t1 = datetime.datetime.strptime('2019-02-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.0')  # 起始时间
    # t2 = datetime.datetime.strptime('2019-03-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.0')  # 结束时间
    # sample_time = (t2 - t1).days * 86400 + (t2 - t1).seconds  # 86400=24*60*60

    sample_seconds = np.array(range(10000, 2419200, sample_interval))  # 重采样的时间序列

    np_list = []  # 用于储存新生成的ndarray
    data_t = data_np.transpose(1, 0)
    last_scan = 0

    for sample_second in sample_seconds:
        # 遍历每一个采样时间点
        for i in range(last_scan, second_series.shape[0] - 1):
            # 遍历原始时间点
            if sample_second == second_series[i]:
                # 若采样时间和原始数据时间相同，直接取值
                np_list.append(data_t[i])
                last_scan = i
                break

            elif second_series[i] < sample_second < second_series[i + 1]:
                # 确定采样时间点所在区间后，用线性插值算出对应时间点的tensor
                temp_np = (sample_second - second_series[i]) / (second_series[i + 1] - second_series[i]) * \
                          (data_t[i + 1] - data_t[i]) + data_t[i]

                np_list.append(temp_np)
                last_scan = i
                break

    data_t = np.array(np_list)  # 将ndarray列表整合成一个ndarray
    sample_np = data_t.transpose(1, 0)

    return sample_seconds, sample_np


if __name__ == '__main__':

    """
    链接数据库，读取数据
    """
    # model = ProcessModel('KX02_YSSC_1006_201902', 'P0X1006WW14, P0X1006WW16, P0X1006WW17, P0X1006WW18, P0X1006WW19, '
    #                                               'P0X1006WW20, P0X1006WW23, P0X1006WW24, P0X1006WW25, P0X1006WW26, '
    #                                               'P0X1006WW27, P0X1006WW28, P0X1006WW29, P0X1006WW30, P0X1006WW31, '
    #                                               'P0X1006WW32, P0X1006WW33, P0X1006WW34, P0X1006WW35, P0X1006WW36')
    #
    # model.connect_database()
    # time_list, data_np = model.read_data()
    # seconds_series = model.generate_time_series(time_list, return_option='seconds_series')
    #
    # np.save('../numpy/1006.npy', data_np)
    # np.save('../numpy/1006_time.npy', seconds_series)

    """
    重采样
    """
    data_np = np.load('../numpy/1006.npy')
    seconds_series = np.load('../numpy/1006_time.npy')
    sample_seconds, sample_np = re_sample(seconds_series[:], data_np[:, :])

    np.save('../numpy/sample_1006.npy', sample_np)
    np.save('../numpy/sample_time.npy', sample_seconds)
