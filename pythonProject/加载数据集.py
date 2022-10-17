# definition Epoch: One forward pass and one backward pass of all the training example.
# definition Batch-Size: The number of training example in one forward backward pass.
# definition Iteration: Number of passes, each pass using [batch size] number of examples
import numpy as np

import torch
from torch.utils.data import Dataset  # 是一个抽象类，不能实例化，只能被继承
from torch.utils.data import DataLoader  # 加载数据


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  # 支持下标操作
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# dataset = DiabetesDataset('../dataset/diabetes.csv.gz')
# num_workers 读mini-batch的时候需要几个线程
# train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


dataset = DiabetesDataset('../dataset/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # enumerate 可以获得当前迭代的次数
            inputs, labels = data

            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
