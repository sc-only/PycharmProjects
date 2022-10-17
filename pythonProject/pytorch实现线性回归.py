import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):  # 通过Module构造出来的对象，会自动根据计算图实现backward的过程
    # 初始化对象时默认执行的函数
    def __init__(self):
        super(LinearModel, self).__init__()  # 不用管，这么写就完事了 just do it :)
        self.linear = torch.nn.Linear(1, 1)  # 构造一个对象，(1, 1) 一个是输入的特征维度，一个是输出的特征维度

    # 必须实现，在前馈的过程中需要执行的计算
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()  # 可callable
# 也是继承至nn.module,参数：size_average：是否求均值,reduce：最终是否要求和，将维度降下来
criterion = torch.nn.MSELoss(size_average=False)  # 计算损失函数,返回值是(y_pred,y)
# 优化器 model.parameters()：取得model中的成员里有相应的权重，就都加到训练的集合中, lr: 学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train Cycle
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)  # 计算损失
    print(epoch, loss)

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向转播
    optimizer.step()  # 进行更新

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)
