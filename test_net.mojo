from net.tensor import Tensor, shape, batch_matmul, matmul
from tensor import Tensor as _Tensor

fn main():
    test_matrix_multipication()

fn test_matrix_multipication():
    var b = 3
    var m = 3
    var n = 2
    var p = 4
    var A = Tensor[DType.int8](b,m,n)
    var B = Tensor[DType.int8](b,n,p)
    A.rand()
    B.rand()
    print(A)
    print(B)
    print(batch_matmul(A,B))