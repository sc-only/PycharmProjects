# 卷积层：保留图像的空间特征
# SubSample 为了减少Feature maps中的特征数量
# 全连接：网络内用的全是线性层，如果网络全部是由线性层串行地连接起来，则将网络称为全连接网络
# 全连接层：每一个输入节点都要参与到下一层输出节点的计算上
# 下采样：通道数不变，减少数据量，降低运算需求，
# 每一个通道都要配一个卷积核，即一个 3*5*5的图像和一个3*3*3的卷积核计算可以得到一个1*3*3的结果
# 如果要得到多个通道的结果，需要将图片通过多个卷积核进行计算
# padding: 可以对卷积结果的size进行调整 BV1Y7411d7Ys 45:32

import torch
in_channels, out_channels = 5, 10  # 输入的通道数量和输出的通道数量
width, height = 100, 100  # 图像的大小
kernel_size = 3  # 卷积核的大小
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)

conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)  # 输入3默认就是3*3，或者输入{5,3}

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
