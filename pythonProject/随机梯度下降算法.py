import numpy as np
import matplotlib
from visdom import Visdom

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

# y=wx
def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

print('Predict (before training', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y) #对每一个样本进行求梯度，然后就进行更新 
        w -= 0.01 * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)

print('Predict (after training)', 4, forward(4))
# plt.plot(w_list,mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()
# if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
