from net.tensor import Tensor
from algorithm import vectorize, parallelize
from sys.info import num_physical_cores, num_logical_cores
from algorithm import Static2DTileUnitFunc as Tile2DFunc
import math


@always_inline
fn tensor_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs an element-wise operation on two tensors of equal shape.

    Parameters:
        dtype : DType of the Tensor.
        func  : (fn[dtype: DType, nelts: Int] (x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts]): The function that performs the element-wise operation.

    Args:
        t1 : Tensor[dtype] The first tensor.
        t2 : Tensor[dtype] The second tensor.

    Returns:
        Returns Tensor[dtype] output tensor.
    """
    var shape = t1.shape == t2.shape
    var elm = t1.num_elements() == t2.num_elements()
    if shape!= elm: 
        print(Error("Both inputs must be the same shape"))
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()
    var res = Tensor[dtype](t1.shape)
    @parameter
    fn calc(i : Int):
        @parameter
        fn vecmath[nelts: Int](idx: Int):
            res.store[nelts](
                idx, func[dtype, nelts](t1.load[nelts](idx), t2.load[nelts](idx))
            )
        vectorize[vecmath, nelts](t1.num_elements())
    parallelize[calc](t1.num_elements(), num_cores)
    return res

@always_inline
fn scalar_op[dtype : DType, func : fn[type: DType, simd_width: Int](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[type, simd_width]](
    Input : Tensor[dtype], value : SIMD[dtype,1]) -> Tensor[dtype]:
    """
    This function performs a scalar operation on a tensor.

    Parameters:
        dtype : DType of the tensor.
        func  : (fn[dtype: DType, simd_width: Int](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]): The function that performs the scalar operation.

    Args:
        Input  : Tensor[dtype] The input tensor.
        value  : SIMD[dtype,1] The scalar value.

    Returns:
        Returns Tensor[dtype] output tensor.
    """
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()
    var Output = Tensor[dtype](Input.shape)

    @parameter
    fn calc(i : Int):

        @parameter
        fn operation[nelts : Int](j : Int):
            Output.store(j, func[dtype,nelts](Input[j], value))
        vectorize[operation, nelts](Input.num_elements())
    parallelize[calc](Input.num_elements(), num_cores)
    
    return Output

@always_inline
fn check_matmul(a: shape, b: shape) -> Bool:
    """
    Checks whether two shapes are compatible for matrix multiplication.

    Args:
        a: The shape of the first tensor.
        b: The shape of the second tensor.

    Returns:
        A Boolean value indicating whether the shapes are compatible for matrix multiplication.
    """
    if a.rank() != b.rank():
        return False

    if a.rank() == 1:
        return a == b

    for i in range(a.rank() - 2):
        if a[i] != b[i]:
            return False

    if a[a.rank() - 2] != b[a.rank() - 1]:
        return False

    return True

fn calculate_shapes(shape1: shape, shape2: shape) -> shape:
    """
    Calculates the resulting shape of the matrix multiplication operation between two input shapes.

    Args:
        shape1: The shape of the first tensor.
        shape2: The shape of the second tensor.

    Returns:
        The resulting shape of the matrix multiplication operation.
    """
    if not check_matmul(shape1, shape2):
        print("Error: Tensors cannot be multiplied due to incompatible shapes.")
        return shape()

    var batch_dims = List[Int]()
    var max_batch_rank = math.max(shape1.rank() - 2, shape2.rank() - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank() - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank() - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print("Error: Batch dimensions do not match and are not broadcastable.")
            print("Error: Incompatible dimensions at index", i, ":", dim1, "vs", dim2)
            return shape()

        batch_dims.append(math.max(dim1, dim2))

    batch_dims.append(shape1[-2])
    batch_dims.append(shape2[-1])

    return shape(batch_dims)

fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)