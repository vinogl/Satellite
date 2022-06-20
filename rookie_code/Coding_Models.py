import torch
from torch import nn


class Coder(nn.Module):

    # 初始化类需传入参数为：原始数据维度、编码后的维度

    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        # 定义编码器模型
        self.encoder = nn.Sequential(
            # 网络需调整，只是暂时写一个
            nn.Softmax(dim=1),
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, encoding_dim),
            nn.Softmax(dim=1)
        )

        # 定义解码器模型
        self.decoder = nn.Sequential(
            # 此处的模型结构和编码器的一致，但传播方向相反
            nn.Softmax(dim=1),
            nn.Linear(encoding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim),  # input_dim = decoding_dim
            nn.Softmax(dim=1)
        )

    def encoder_loss(self, inputs):

        # 对比输入（inputs）、编码（encoded）即可得到损失
        y = self.encoder(inputs)  # y--encoded
        x = nn.Softmax(dim=1)(inputs)  # x--inputs

        # 损失函数的表达式（须优化）
        loss = 0.
        # 迭代计算损失：loss = (1/n^2) * sum(yi*yj-xi*xj)^2
        for i in range(len(inputs)):
            # torch.mul: 张量对应位相乘，这里每个循环用每一列乘所有列
            mul_x_i = torch.mul(x[i], x).sum(dim=1)
            mul_y_i = torch.mul(y[i], y).sum(dim=1)
            loss += nn.MSELoss()(mul_x_i, mul_y_i)

        return loss

    def decoder_loss(self, inputs, encoded):

        # 对比输入数据（inputs）和解码数据（decoded）得到损失函数
        decoded = self.decoder(encoded)
        loss = nn.MSELoss()(inputs, decoded)

        return loss
