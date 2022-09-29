import numpy as np


def dis_abs(x, y):
    return abs(x-y)[0]

def estimate_twf(A, B, dis_func=dis_abs):
    # 向前传播计算每个节点之间的最短距离
    N_A = len(A)
    N_B = len(B)
    D = np.zeros([N_A, N_B])
    D[0, 0] = dis_func(A[0], B[0])

    # 先计算边缘
    for i in range(1, N_A):
        D[i, 0] = D[i-1, 0] + dis_func(A[i], B[0])
    for j in range(1, N_B):
        D[0, i] = D[0, i-1] + dis_func(A[0], B[i])
    # 再计算中间
    for i in range(1, N_A):
        for j in range(1, N_B):
            D[i, j] = dis_func(A[i], B[j]) +min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # 反向传播计算最短路径
    i = N_A - 1
    j = N_B - 1
    count = 0
    d = np.zeros(max(N_A, N_B)*2)  # 路径上的两点之间的距离
    path = []  # 记录经过的所有路径
    while True:
        if i > 0 and j > 0:
            path.append((i, j))
            m = min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            if m == D[i-1, j]:
                d[count] = D[i, j] - D[i-1, j]
                i = i-1
                count = count+1
            elif m == D[i, j-1]:
                d[count] = d[i, j] - D[i, j-1]
                j = j-1
                count = count+1
            elif m == D[i-1, j-1]:
                d[count] = d[i, j] -D[i-1, j-1]
                i = i-1
                j = j-1
                count = count+1
        elif i == 0 and j == 0:
            path.append((i, j))
            d[count] = D[i, j]
            count = count+1
            break
        elif i == 0:
            path.append((i, j))
            d[count] = d[i, j] - d[i, j-1]
            j = j-1
            count = count+1
        elif j == 0:
            path.append((i,j))
            d[count] = d[i, j] - d[i-1, j]
            i = i-1
            count = count+1

    mean = np.sum(d) / count
    return mean, path[::-1], D  # path[::-1]表示数组取反

