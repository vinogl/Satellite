import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.siamese import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils_fit import fit_one_epoch


def get_image_num(path):
    """
    计算图片总数
    """
    num = 0
    train_path = path

    for character in os.listdir(train_path):
        # 在大众类下遍历小种类。
        character_path = os.path.join(train_path, character)
        num += len(os.listdir(character_path))
    return num


if __name__ == "__main__":

    # 是否使用Cuda
    Cuda = True

    # 训练的数据类型
    pic_type = 'Markov'

    # 待训练数据集存放的路径
    dataset_path = '../pic/%s' % pic_type

    # 训练后，历史损失的保存地址
    save_path = r'../numpy/%s' % pic_type

    # 训练log保存地址
    log_path = 'logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # 输入图像的大小，默认为105,105,3
    input_shape = [105, 105, 3]

    # 用于指定是否使用VGG预训练权重
    pretrained = False
    model_path = ""

    """
    此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    如果想要让模型从0开始训练，则设置model_path = ''，pretrain = False，此时从0开始训练。
    一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    """

    model = Siamese(input_shape, pretrained)
    if model_path != '':
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = nn.BCELoss()

    # 训练集和验证集的比例。
    train_ratio = 0.9
    images_num = get_image_num(dataset_path)
    num_train = int(images_num * train_ratio)
    num_val = images_num - num_train

    """
    训练分为两个阶段，两阶段初始的学习率不同，手动调节了学习率
    显存不足与数据集大小无关，提示显存不足请调小batch_size。
    """

    # 用于存储训练过程中的历史损失
    total_loss_np = np.array([])
    val_loss_np = np.array([])

    if True:
        Batch_size = 32
        Lr = 1e-4
        Init_epoch = 0
        Freeze_epoch = 50

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Init_epoch, Freeze_epoch):
            total_loss, val_loss = fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val,
                                                 gen, gen_val, Freeze_epoch, Cuda, log_path)
            lr_scheduler.step()

            total_loss_np = np.append(total_loss_np, total_loss)
            val_loss_np = np.append(val_loss_np, val_loss)

    if True:
        Batch_size = 20
        Lr = 1e-5
        Freeze_epoch = 50
        Unfreeze_epoch = 100

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)

        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Freeze_epoch, Unfreeze_epoch):
            total_loss, val_loss = fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val,
                                                 gen, gen_val, Unfreeze_epoch, Cuda, log_path)
            lr_scheduler.step()

            total_loss_np = np.append(total_loss_np, total_loss)
            val_loss_np = np.append(val_loss_np, val_loss)

    # 保存训练过程损失数据
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, 'total_loss.npy'), total_loss_np)
    np.save(os.path.join(save_path, 'val_loss.npy'), val_loss_np)
