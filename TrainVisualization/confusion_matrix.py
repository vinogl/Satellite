import numpy as np
from matplotlib import pyplot as plt


np_type_list = ['Markov', 'Gramian', 'Recurrence']

# 设定正负样本阀值
positive_threshold = 0.85
negative_threshold = 0.15

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

for np_type in np_type_list:
    plt.suptitle({'Markov': 'MTF', 'Gramian': 'GAF', 'Recurrence': 'RP'}[np_type])
    confusion = np.zeros([4, 4])  # 用于存储设定样本阀值后的混淆矩阵数值
    for i in range(4):
        for j in range(4):
            temp_np = np.load('../numpy/%s/CM/%d-%d.npy'
                              % (np_type, i, j))

            plt.subplot(4, 4, i * 4 + j + 1)
            if i == j:
                rate = np.sum(temp_np > positive_threshold) / temp_np.shape[0]
                confusion[i, j] = int(rate * 10000)/100  # 混淆矩阵中保留两位小数

            else:
                rate = np.sum(temp_np < negative_threshold) / temp_np.shape[0]
                confusion[i, j] = int(10000 - rate * 10000)/100  # 混淆矩阵中保留两位小数

            plt.xticks([])
            plt.yticks([0, 1])
            plt.ylim((0, 1))
            plt.scatter(range(temp_np.shape[0]), temp_np, marker='.')

    plt.savefig('../pic/confusion_matrix/%s.png' % np_type, transparent=True)
    plt.clf()

    # 画混淆矩阵
    # 热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar()

    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    indices = range(len(confusion))
    classes = ['normal', 'fault_1', 'fault_2', 'fault_3']
    plt.xticks(indices, classes, rotation=0)  # 设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.

    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(second_index, first_index, confusion[first_index][second_index],
                     horizontalalignment="center",
                     color="white" if confusion[first_index, second_index] > thresh else "black")

    plt.savefig('../pic/confusion_matrix/%s_num.png' % np_type, transparent=True)
    plt.clf()
