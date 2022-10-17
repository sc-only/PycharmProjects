import torchvision

# 常用的数字分类数据集
train_set = torchvision.datasets.MNIST(root='~/PycharmProjects/dataset', train=True, download=True)
train_set = torchvision.datasets.MNIST(root='~/PycharmProjects/dataset', train=False, download=True)

# 一组32*32的彩色小图片，分为10类
train_set = torchvision.datasets.CIFAR10(root='~/PycharmProjects/dataset', train=True, download=True)
train_set = torchvision.datasets.CIFAR10(root='~/PycharmProjects/dataset', train=False, download=True)

