import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.siamese import Siamese as siamese


class Siamese(object):
    """
    此文件的Siamese模型用于my_predict.py中预测
    """
    def __init__(self, model_path, input_shape, cuda):
        self.model_path = model_path
        self.input_shape = input_shape
        self.cuda = cuda
        self.generate()

    # 载入模型
    def generate(self):
        """
        载入模型与权值
        """
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    def detect_image(self, image_1, image_2):
        """
        检测图片
        """
        # 对输入图像进行不失真的resize
        image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
        image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

        # 对输入图像进行归一化
        photo_1 = np.asarray(image_1).astype(np.float64) / 255
        photo_2 = np.asarray(image_2).astype(np.float64) / 255

        if self.input_shape[-1] == 1:
            photo_1 = np.expand_dims(photo_1, -1)
            photo_2 = np.expand_dims(photo_2, -1)

        with torch.no_grad():

            # 添加上batch维度，才可以放入网络中预测
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # 获得预测结果，output输出为概率
            output = self.net([photo_1, photo_2])[0]
            output = torch.nn.Sigmoid()(output)

        return output
