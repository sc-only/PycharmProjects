# padding: 可以对卷积结果的size进行调整 BV1Y7411d7Ys 45:32

import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)  # 输出通道数，输入通道数，宽度，高度
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output.shape)
print(output)

