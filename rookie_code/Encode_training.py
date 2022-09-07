import torch
import Coding_Models
from data_read import read_csv


"""
保存参数格式: torch.save(模型.state_dict(), 路径)
读取参数格式: 模型.load_state_dict(torch.load(路径))
"""
encoder_parameter_save_path = ''  # 训练好后的模型参数


def encoder():
    """读取编码器模型的参数，供调用"""
    return model.encoder.load_state_dict(torch.load(encoder_parameter_save_path))


if __name__ == '__main__':
    # 数据导入
    data_path = '/Users/GongLi/Desktop/好雨+龚理/2017-01-01.csv'
    _, _, input_data = read_csv(data_path,  data_start_row=2, data_start_column=1, time_column=0)

    # 超参数设定
    encoding_dim = 40  # 编码器的输出维度
    model = Coding_Models.Coder(input_dim=input_data.shape[1], encoding_dim=encoding_dim)
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 32
    data_loader = torch.utils.data.DataLoader(input_data, batch_size=batch_size, shuffle=True)
    epochs = 100

    # 开始训练
    for epoch in range(1, epochs + 1):
        for loader in data_loader:
            loss = model.encoder_loss(loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印损失
        if epoch % 10 == 0:
            print('loss: %f' % float(loss))

    # 保存编码器模型的参数
    torch.save(model.encoder.state_dict(), encoder_parameter_save_path)
