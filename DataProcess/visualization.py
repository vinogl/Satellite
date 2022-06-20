import numpy as np
from matplotlib import pyplot as plt
from pyts.image import MarkovTransitionField, GramianAngularField, RecurrencePlot
from progress.bar import IncrementalBar
import warnings


warnings.filterwarnings('ignore')  # 抑制警告显示

tran_type_list = ['Markov', 'Gramian', 'Recurrence']
data_type_list = ['normal', 'fault_1', 'fault_2', 'fault_3']

cycle = 567  # 数组周期(以10s为采样间隔的数据点数)

for tran_type in tran_type_list:
    for data_type in data_type_list:

        bar = IncrementalBar("%s-%s" % (tran_type, data_type), suffix='%(percent)d%%')  # 用作显示运行进度

        data = np.load('../numpy/V-A/%s.npy' % data_type)

        for i in bar.iter(range(int(data.shape[0] / cycle))):

            if tran_type == 'Markov':
                function = MarkovTransitionField()
            elif tran_type == 'Gramian':
                function = GramianAngularField(method='summation')
            elif tran_type == 'Recurrence':
                function = RecurrencePlot(dimension=3, time_delay=10)

            X_tran = function.fit_transform([data[i * cycle:(i + 1) * cycle]])

            # Show the image for the first time series
            plt.axis('off')
            plt.imshow(X_tran[0], cmap='rainbow', origin='lower')
            plt.tight_layout()

            plt.savefig('../pic/%s/%s/%d-%d.jpg'
                        % (tran_type, data_type, i, i+1), bbox_inches='tight', pad_inches=0)
            plt.clf()  # 更新画布
