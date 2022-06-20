import torch
import csv
import datetime


def read_csv(data_path, data_start_row=2, data_start_column=1, time_column=0):
    """
    仅用于此项目的csv文件阅读
    输入为：文件路径，数据的起始行、列，时间列，计数列
    输出为：表的抬头（headers）、时间的行映射（time_dic）、tensor类型的数据（data_tensor）
    当time_column=counting_column时，
    """
    with open(data_path, encoding='utf-8') as f:  # encoding: utf-8、GB2312、gbk、ISO-8859-1
        # f_csv = csv.reader(_.replace('\x00', '') for _ in f)
        f_csv = csv.reader(f)

        header_dic = {i: header for i, header in enumerate(next(f_csv))}  # csv的表头保存为headers
        for i in range(data_start_row-1):
            next(f_csv)
        time_dic = {}  # 时间对应数据行的映射
        data_list = []  # 各行数据

        for i, row in enumerate(f_csv):
            dic_len_before = len(time_dic)  # 新增元素前的词典长度
            time_i = datetime.datetime.strptime(row[time_column], '%Y-%m-%d %H:%M:%S.0')
            time_dic.update({time_i: i})
            dic_len_after = len(time_dic)  # 新增元素后的词典长度

            if dic_len_before != dic_len_after:  # 可避免重复时间点的数据录入
                row_float = [float(string) for string in row[data_start_column:]]  # 将string转为float
                data_list.append(row_float)

        data_tensor = torch.tensor(data_list)

        time_dic = dict(zip(time_dic.values(), time_dic.keys()))

    return header_dic, time_dic, data_tensor


def generate_time_series(time_dic, return_option='time_series'):
    """
    根据read_csv生成的time_dic，得到包含时刻间隔的时间序列
    格式必须完全符合time_dic，即：{序号：时间}
    """
    zero_time = '2019-02-01 00:00:00.0'  # 设定时间的零点
    ex_sec = datetime.datetime.strptime(zero_time, '%Y-%m-%d %H:%M:%S.0')  # 字符串转换成时间

    seconds_series = []  # 计数点秒数
    seconds_dic = {}  # 计数点与秒数的映射
    diff_series = []  # 计数间秒间隔序列
    time_sec = 0
    for key, val in time_dic.items():
        sec = val
        diff = float((sec - ex_sec).days * 86400 + (sec - ex_sec).seconds)  # 计算该时刻与前一时刻的时间差
        ex_sec = sec  # 本时刻时间赋值给下一时刻的前一刻时间
        time_sec += diff

        seconds_series.append(time_sec)  # 将每个时刻加入时间序列
        seconds_dic.update({key: time_sec})  # 每个时刻与计数映射
        diff_series.append(diff)

    if return_option == 'seconds_series':
        return torch.tensor(seconds_series)
    elif return_option == 'seconds_dic':
        return seconds_dic
    elif return_option == 'diff_series':
        return torch.tensor(diff_series)
    else:
        exit('return_option只可选times_series、series_dic、diff_series')


def data_supplement(time_dic, data_ts):
    """
    数据填充：
    当序列中某时刻数据为0，函数将基于左右不为0的数据对该点线性插值
    """
    secs = generate_time_series(time_dic, return_option='seconds_series')
    for i, data in enumerate(data_ts):
        for j, _ in enumerate(data):
            if data[j] == 0:
                # 找到下一个非0点的位置
                for k in range(1, 100):
                    if data[j + k] != 0:
                        break
                # 插值法补值
                data_ts[i, j] = ((secs[j] - secs[j - 1]) / (secs[j + k] - secs[j - 1])) * \
                                    (data[j + k] - data[j - 1]) + data[j - 1]
    return data_ts


def data_preprocess(data_path):
    """
    对数据预处理：
    read_csv中剔除重复数据
    data_supplement中补齐缺失数据
    generate_time_series根据给定的时间零点，生成相应的秒序列
    """
    header_dic, time_dic, data_tensor = read_csv(data_path, 2, 1, 0)

    data_ts = data_supplement(time_dic, data_tensor.permute(1, 0))  # 补齐缺失数据
    second_series = generate_time_series(time_dic, return_option='seconds_series')  # 与数据对应的时间序列

    return second_series, data_ts, header_dic


def re_sample(second_series, data_ts, sample_interval=60):
    """
    根据给定的采样间隔，对原始数据重采样。
    不同表的采样间隔不同，用此方法，可将不同表的数据统一处理。
    """
    t1 = datetime.datetime.strptime('2019-02-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.0')  # 起始时间
    t2 = datetime.datetime.strptime('2019-03-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.0')  # 结束时间
    sample_time = (t2 - t1).days * 86400 + (t2 - t1).seconds  # 86400=24*60*60
    sample_seconds = torch.tensor(range(0, sample_time, sample_interval))  # 重采样后对应的时间序列

    tensor_list = []  # 用于储存新生成的tensor
    data_tensor = data_ts.permute(1, 0)

    for sample_second in sample_seconds:
        # 遍历每一个采样时间点
        for i in range(second_series.shape[0]-1):
            # 遍历原始时间点
            if sample_second == second_series[i]:
                # 若采样时间和原始数据时间相同，直接取值
                tensor_list.append(data_tensor[i].unsqueeze(0))

            elif second_series[i] < sample_second < second_series[i + 1]:
                # 确定采样时间点所在区间后，用线性插值算出对应时间点的tensor
                temp_tensor = (sample_second - second_series[i]) / (second_series[i + 1] - second_series[i]) * \
                              (data_tensor[i + 1] - data_tensor[i]) + data_tensor[i]

                tensor_list.append(temp_tensor.unsqueeze(0))

    data_tensor = torch.cat(tensor_list, dim=0)  # 将tensor列表整合成一个tensor
    data_ts = data_tensor.permute(1, 0)

    return data_ts
