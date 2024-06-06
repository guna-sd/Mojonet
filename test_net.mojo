from net.tensor import Tensor, shape, batch_matmul, matmul
from tensor import Tensor as _Tensor
import time

fn main():
    test_matrix_multipication()

fn test_matrix_multipication():
    var b = 3
    var m = 1024
    var n = 2048
    var p = 1024
    var A = Tensor[DType.int64](b,m,n)
    var B = Tensor[DType.int64](b,n,p)
    A.rand()
    B.rand()
    print()
    print(benchmark_matmul[DType.int64, batch_matmul](A,B))

fn benchmark_matmul[T : DType, func : fn[type : DType](tensor1 : Tensor[type], tensor2 : Tensor[type]) -> Tensor[type]
    ](tensor1 : Tensor[T], tensor2 : Tensor[T], num_iterations : Int =10) -> Float64:

    var start_time = time.now()
    for _ in range(num_iterations):
        var result = func(tensor1, tensor2)
    var end_time = time.now()
    return (end_time - start_time) / 1e+9