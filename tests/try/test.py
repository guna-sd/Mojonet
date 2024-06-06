import torch
from torch import Tensor
import time
import numpy as np

a = Tensor([[[48, 17], 
        [76, 34], 
        [71, 118]],

        [[36, 102], 
        [55, 93], 
        [105, 23]],

        [[26, 83], 
        [72, 14], 
        [47, 40]]]).type(torch.int8)

b = Tensor([[[49, 67, 74, 67], 
        [83, 123, 22, 34]],

        [[92, 35, 64, 4], 
        [43, 28, 15, 84]],

        [[9, 75, 120, 61], 
        [72, 38, 22, 94]]]).type(torch.int8)

c = Tensor([[[-77, -69, 86, -46], 
        [-110, 58, -28, 104], 
        [-39, 71, -86, 65]],

        [[18, 20, -6, 8], 
        [99, -79, 51, 96], 
        [-103, -33, -103, 48]],

        [[66, -16, 82, -84], 
        [120, 44, -12, 76], 
        [-25, -75, 120, -29]]]).type(torch.int8)
# print(a.bmm(b))
# print(np.matmul(a.tolist(), b.tolist()))
# print(c)
# print((a.bmm(b)) == c)
#    print(__type_of(c).__str__(c))

def benchmark_matmul(tensor1, tensor2, func, num_iterations=10):
    start_time = time.time()
    for _ in range(num_iterations):
        result = func(tensor1, tensor2)
    end_time = time.time()
    return end_time - start_time

b = 3
m = 1024
n = 2048
p = 1024

a = torch.rand(b, m, n).type(torch.int64)
b = torch.rand(b, n, p).type(torch.int64)
print()
print(benchmark_matmul(a, b, torch.bmm))
