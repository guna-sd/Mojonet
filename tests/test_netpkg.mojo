from net import Tensor
import net
import time


alias b = 1
alias m = 1024
alias n = 2048
alias p = 1024


fn bench():
    var A = Tensor[DType.float32](b,m,n).random()
    var B = Tensor[DType.float32](b,n,p).random()
    benchmark_matmul(A, B)


fn benchmark_matmul[
    T: DType,
](tensor1: Tensor[T], tensor2: Tensor[T], num_iterations: Int = 1):
    print("start")
    var start_time = time.now()
    for _ in range(num_iterations):
        _ = ((tensor1@tensor2))
    var end_time = time.now()
    print((end_time - start_time) / 1e9)

fn main():
    bench()