import numpy as np
import matplotlib
from visdom import Visdom
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)
# y=wx+b
w_list = [] #权重
b_list = []
mse_list = [] #对应权重的损失值
for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2,2.1,0.1):
        print('w=',w)
        print('b=',b)
        l_sum=0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=loss(x_val,y_val)
            l_sum+=loss_val
            print('\t',x_val,y_val,y_pred_val,loss_val)
        print('MSE=',l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)

# plt.plot(w_list,mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()
W, B = np.meshgrid(w_list, b_list)
ax3 = plt.axes(projection='3d')
mse_list=np.array(mse_list)
ax3.plot_surface(W,B,mse_list,cmap='rainbow')
plt.show()
# if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
