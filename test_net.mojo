from net.tensor import Tensor, shape
import math
from net.kernel import calculate_shapes, matmul
from testing import assert_equal
from tensor import Tensor as _Tensor
from net.kernel import randn, matmul2d
from net.checkpoint import File
import time
import os
from sys import exit

fn main() raises:
    # var tensor1 = Tensor[DType.int16](2,4)
    # var nshape = shape(List[Int](1,4))
    # var tensor2 = Tensor[DType.int16](nshape)
    # var tensor3 = net.ones[DType.bfloat16](2,4)
    #test_matrix_multipication()
    # var random = randn()
    # var randnum = random.randf16(0,1)
    # # print(time.now() // 10**9)
    
    # print(randnum)
    # print(simdwidthof[DType.int8]())
    # print(sizeof[DType.int8]())
    # var testa = Tensor(4,4).random()
    # var testb = Tensor(4,4).random()
    # print(testa)
    # print(testa[0][2])
    # #print(round(11.319262504577637, 4))
    # print(testb)
    # print(matmul2d[DType.float32](testa, testb))
    # # var a = Tensor[DType.bfloat16](4,5,4)
    # # a.rand()
    # # print(a)
    var a = Tensor(1,1,5)
    print(a.random())
    var b = Tensor(1,4,5).random()
    print(b)
    print(a + b)
    # var s1 = List[Int](4,5)
    # var s2 = List[Int](1,3,5)
    # var s3 = calculate_indices(s1, 5)
    # print(__type_of(s3).__str__(s3))

fn calculate_broadcast_shape(shape1: List[Int], shape2: List[Int]) -> List[Int]:
    var max_dim = math.max(shape1.__len__(), shape2.__len__())
    var broadcasted_shape: List[Int] = List[Int]()
    
    for i in range(max_dim):
        var dim1 = shape1[-(i + 1)] if i < shape1.__len__() else 1
        var dim2 = shape2[-(i + 1)] if i < shape2.__len__() else 1
        
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print("Shapes are not compatible for broadcasting: [", __type_of(shape1).__str__(shape1), "] and [", __type_of(shape2).__str__(shape2), "]")
            exit()
        
        broadcasted_shape.insert(0, math.max(dim1, dim2))
    
    return broadcasted_shape

@always_inline
fn calculate_indices(shape: List[Int], index: Int) -> List[Int]:
    """
    Converts a linear index into its corresponding multi-dimensional indices based on the given shape.

    This function is useful for determining the multi-dimensional indices of an element in a tensor or array,
    given its linear index (i.e., its position in a flattened version of the tensor or array) and the shape of the tensor or array.

    Args:
        shape: A List[Int] representing the dimensions of the tensor or array.
        index: An Int representing the linear index of an element in the flattened tensor or array.

    Returns:
        A List[Int] containing the multi-dimensional indices corresponding to the given linear index.
    """
    var dim_indices = List[Int]()
    var num_dims = len(shape)
    var linear_index = index
    for i in range(num_dims - 1, -1, -1):
        var dim_size = shape[i]
        var dim_index = linear_index % dim_size
        dim_indices.append(dim_index)       
        linear_index //= dim_size
    dim_indices.reverse()
    return dim_indices


fn round(number : Float64, ndigits : Int)-> Float64:
    """
    Rounds a floating-point number to a specified number of decimal places.
    
    :param number: The number to be rounded.
    :param ndigits: The number of decimal places to round to. If None, round to the nearest integer.
    :return: The rounded number.
    """
    var factor = 10 ** ndigits
    return int(number * factor + 0.5 if number > 0 else number * factor - 0.5) / factor

fn test_reshape_basic() raises:
  var shapes = shape(List[Int](3,2))
  var tensor = Tensor[DType.int8](shapes)
  tensor.rand()
  print(tensor)
  var reshaped_tensor = tensor.reshape(shape(2,3))
  assert_equal (reshaped_tensor.shape , shape(2, 3))
  print(reshaped_tensor)
  print(reshaped_tensor.transposed())

def test_div():
    # Create two tensors with different shapes
    tensor1 = Tensor[DType.bfloat16](3,1,5)
    tensor2 = Tensor[DType.bfloat16](1,4,5)
    tensor1.rand(4)
    tensor2.rand(4)
    print(tensor1)
    print(tensor2)

    print(tensor1.multiply(tensor2))


fn test_matrix_multipication():
    var A = Tensor[DType.bfloat16](1024, 1024)
    var B = Tensor[DType.bfloat16](1024, 2056)
    A.rand()
    B.rand()
    print(A)
    print(B)
    print(matmul[DType.bfloat16](A,B))
    print(A@B)