from net.tensor import Tensor, shape
import net as torch
from net.nn.activation import Fuctional as F
import math
from net.kernel import matmul_submatrix, calculate_shapes, accumulate, matmul
from testing import assert_equal
from tensor import Tensor as _Tensor
from net.kernel import randn
import time
fn main():
    # var tensor1 = Tensor[DType.int16](2,4)
    # var nshape = shape(List[Int](1,4))
    # var tensor2 = Tensor[DType.int16](nshape)
    # var tensor3 = net.ones[DType.bfloat16](2,4)
    #test_matrix_multipication()
    var random = randn()
    var randnum = random.randf64(0,1)
    # print(time.now() // 10**9)
    print(randnum)
    # var a = Tensor[DType.bfloat16](4,5,4)
    # a.rand()
    # print(a)

fn test_reshape_basic() raises:
  var shapes = shape(List[Int](3,2))
  var tensor = Tensor[DType.int8](shapes)
  tensor.rand()
  print(tensor)
  var reshaped_tensor = tensor.reshape(shape(2,3))
  assert_equal (reshaped_tensor.shape , shape(2, 3))
  print(reshaped_tensor)
  print(reshaped_tensor.transposed())


fn test_matrix_multipication():
    var A = Tensor[DType.bfloat16](1024, 1024)
    var B = Tensor[DType.bfloat16](1024, 2056)
    A.rand()
    B.rand()
    print(A)
    print(B)
    print(matmul[DType.bfloat16](A,B))
    print(A@B)