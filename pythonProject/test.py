import numpy as np

a = np.array([[1, 0, 2, 0, 0, 0],
              [0, 1, 0, 2, 0, 0],
              [0, 0, 1, 0, 2, 0],
              [0, 0, 0, 1, 0, 2]])
b=a.T
print(np.matmul(a, b))