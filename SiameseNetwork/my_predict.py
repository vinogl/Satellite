import os
import numpy as np
from siamese import Siamese
from PIL import Image


pic_type_list = ['Markov', 'Gramian', 'Recurrence']

# 各分类的文件名
class_file = {0: 'normal', 1: 'fault_1', 2: 'fault_2', 3: 'fault_3'}

# 各分类的标准样本(用其他样本与该样本作对比的相似度作为分类准确度)
class_sample = {0: 'normal/0-1.jpg', 1: 'fault_1/0-1.jpg', 2: 'fault_2/0-1.jpg', 3: 'fault_3/0-1.jpg'}


for pic_type in pic_type_list:

    data_path = r'../pic/%s' % pic_type
    model = Siamese(model_path='model_data/%s.pth' % pic_type, input_shape=(105, 105, 3), cuda=True)

    for real, file_path in class_file.items():
        pic_list = os.listdir(os.path.join(data_path, file_path))

        for predict, sample in class_sample.items():
            np_temp = np.array([])

            for pic in pic_list:

                image_1 = Image.open(os.path.join(data_path, sample))
                image_2 = Image.open(os.path.join(os.path.join(data_path, file_path), pic))
                probability = model.detect_image(image_1, image_2)
                np_temp = np.append(np_temp, probability.cpu())

            save_path = r'../numpy/%s/CM' % pic_type

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            np.save(os.path.join(save_path, '%d-%d.npy' % (real, predict)), np_temp)
