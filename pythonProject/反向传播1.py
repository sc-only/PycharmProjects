import numpy as np
import matplotlib
import torch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w1.requires_grad = True

# y=w*x
def forward(x):
    return x * w1


# 构建计算图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print('Predict (before training', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 计算图上所有要计算梯度的地方，把梯度都求出来
        print('\tgrad:', x, y, w1.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data  # 如果直接计算grad 会构建计算图

        w1.grad.data.zero_()  # 将权重中的梯度数据全部清零
    print("progress:", epoch, l.item())
print('w1=', w1.item())
print('Predict (after training)', 4, forward(4).item())
# plt.plot(w_list,mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()
# if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
