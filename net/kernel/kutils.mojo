from net.tensor import Tensor
from algorithm import vectorize, parallelize
from sys.info import num_physical_cores, num_logical_cores
import math
from collections.optional import Optional
import time.time as time


@always_inline
fn tensor_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
            t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
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
        print(Error("Tensors must be in same shape"))
        abort(external_call["exit", Int](1))
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()-2 if num_physical_cores() > 4 else 2
    var res = Tensor[dtype](t1.shape)

    @parameter
    fn calc(i : Int):
        @parameter
        fn operation[nelts: Int](idx: Int):
            res[idx] = func(t1.load(idx), t2.load(idx))
        vectorize[operation, nelts, unroll_factor=4](t1.num_elements())
    parallelize[calc](t1.num_elements(), num_cores)

    return res

@always_inline
fn scalar_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
    Input : Tensor[dtype], value : SIMD[dtype,1]) -> Tensor[dtype]:
    """
    This function performs a scalar operation on a tensor.

    Parameters:
        dtype : DType of the tensor.
        func  : (fn[dtype: DType, simd_width: Int](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]): The function that performs the scalar operation.

    Args:
        Input  : Tensor[dtype] The input tensor.
        value  : Scalar[dtype] The scalar value.

    Returns:
        Returns Tensor[dtype] output tensor.
    """
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()-2 if num_physical_cores() > 4 else 2
    var Output = Tensor[dtype](Input.shape)

    @parameter
    fn calc(i : Int):
        @parameter
        fn operation[nelts : Int](j : Int):
            Output[j] = func(Input[j], value)
        vectorize[operation, nelts, unroll_factor=4](Input.num_elements())
    parallelize[calc](Input.num_elements(), num_cores)
    
    return Output

fn Broadcast_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
            t1 : Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:

    var broadcasted_shape = t1.shape.broadcast_shapes(t2.shape)
    if broadcasted_shape.rank() == 0:
        print(Error("Cannot Broadcast tensor with incompatible shapes"))
        abort(external_call["exit", Int](1))

    var result = Tensor[dtype](broadcasted_shape)
    alias nelts = simdwidthof[dtype]()
    var num_elements = result.num_elements()
    var num_cores = num_physical_cores()-2 if num_physical_cores() > 4 else 2

    @parameter
    fn calc(start_index: Int):
        @parameter
        fn operation[nelts : Int](index: Int):
            var result_indices = result.shape.indices(start_index + index)
            var other_indices = t2.shape.indices(start_index + index)
            for j in range(t1.shape.rank()):
                if t1.shape[j] == 1:
                    result_indices[j] = 0
            for j in range(t2.shape.rank()):
                if t2.shape[j] == 1:
                    other_indices[j] = 0
            result[start_index + index] = func(t1[result_indices], t2[other_indices])
        vectorize[operation, nelts, unroll_factor=4](num_elements - start_index)
    parallelize[calc](num_elements, num_cores)

    return result

@always_inline
fn check_shape(a: shape, b: shape) -> Bool:
    """
    Checks whether two shapes are compatible for matrix multiplication.

    Args:
        a: The shape of the first tensor.
        b: The shape of the second tensor.

    Returns:
        A Boolean value indicating whether the shapes are compatible for matrix multiplication.
    """
    if a.rank() < 1 or b.rank() < 1:
        return False

    if a.rank() == 1 and b.rank() == 1:
        return a[0] == b[0]
    
    if a.rank() == 1 and b.rank() > 1:
        return a[0] == b[b.rank() - 2]

    if b.rank() == 1 and a.rank() > 1:
        return a[a.rank() - 1] == b[0]
    
    if a[-1] != b[-2]:
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
    if not check_shape(shape1, shape2):
        print("Error: Tensors cannot be multiplied due to incompatible shapes.")
        abort(external_call["exit", Int](1))

    var batch_dims = List[Int]()
    var max_batch_rank = math.max(shape1.rank() - 2, shape2.rank() - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank() - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank() - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print("Error: Incompatible dimensions at index", i, ":", dim1, "vs", dim2)
            print("Error: Batch dimensions do not match and are not broadcastable.")
            abort(external_call["exit", Int](1))

        batch_dims.append(math.max(dim1, dim2))

    if shape1.rank() > 1 and shape2.rank() > 1:
        batch_dims.append(shape1[shape1.rank() - 2])
        batch_dims.append(shape2[shape2.rank() - 1])
    elif shape1.rank() > 1 and shape2.rank() == 1:
        batch_dims.append(shape1[shape1.rank() - 2])
    elif shape1.rank() ==1 and shape2.rank() > 1:
        batch_dims.append(shape2[shape2.rank() - 1]) 

    return shape(batch_dims)


@always_inline
fn compute_matrix_block[dtype : DType,](
    result: DTypePointer[dtype],
    matrix1: DTypePointer[dtype],
    matrix2: DTypePointer[dtype],
    MatrixHeight: Int,
    MatrixWidth: Int,
    MatrixDepth: Int,
    BlockHeight: Int, BlockWidth: Int,
    block_row_index: Int,
    block_column_index: Int,
):
    # Compute tile
    alias nelts = simdwidthof[dtype]()
    var accumulator = DTypePointer[dtype](BlockHeight * BlockWidth)
    memset_zero[dtype](accumulator, BlockHeight * BlockWidth)

    for depth_index in range(MatrixDepth):
        @unroll
        for block_row in range(BlockHeight):
            @parameter
            fn process_block_column[nelts: Int](block_column: Int):
                accumulator.store[width=nelts](
                    block_row * BlockWidth + block_column,
                    SIMD[dtype, nelts].splat(matrix1[(block_row_index + block_row) * MatrixDepth + depth_index])
                    .fma(
                        matrix2.load[width=nelts](depth_index * MatrixWidth + (block_column_index + block_column)),
                        accumulator.load[width=nelts](block_row * BlockWidth + block_column),),)
            vectorize[process_block_column, nelts](BlockWidth)

    # Store tile
    for block_row in range(BlockHeight):

        @parameter
        fn store_block_column[nelts: Int](block_column: Int):
            result.store[width=nelts](
                (block_row_index + block_row) * MatrixWidth + (block_column_index + block_column),
                accumulator.load[width=nelts](block_row * BlockWidth + block_column)
            )

        vectorize[store_block_column, nelts](BlockWidth)


struct randn:
    alias Int8MAX = Int8.MAX
    alias Int16MAX = Int16.MAX
    alias Int32MAX = Int32.MAX
    alias Int64MAX = Int64.MAX
    alias F16Max : Float16 = 65504.0
    alias F32Max : Float32 = 3.4028234663852886e+38
    alias F64Max : Float64 = 1.7976931348623157e+308

    var _seed : Optional[Int]

    fn __init__(inout self):
        self._seed = time.now()

    fn seed(inout self):
        """
        Seeds the random number generator using the current time.
        """
        self._seed = time.now()
    
    fn seed(inout self, seed : Int):
        """
        Seeds the random number generator with the specified seed.

        Args:
            seed: Int The seed value.
        """
        self._seed = seed
    
    fn lcg(self) -> Int:
        return (self._seed.value()[] * 1103515245 + 12345) % 65504_1234

    fn randint8(self) -> Int:
        """
        Generates a random integer of type Int8.

        Returns:
            A random integer value of type Int8.
        """
        return self.lcg() % int(self.Int8MAX)

    fn randint8(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer between low and high (inclusive) of type Int8.

        Args:
            low: The lower bound (inclusive) of the random integer range.
            high: The upper bound (inclusive) of the random integer range.

        Returns:
            A random integer value of type Int8 between low and high (inclusive).
        """
        return low + self.randint8() % (high - low + 1)

    fn randint16(self) -> Int:
        """
        Generates a random integer of type Int16.

        Returns:
            A random integer value of type Int16.
        """
        return self.lcg() % int(self.Int16MAX)

    fn randint16(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer between low and high (inclusive) of type Int16.

        Args:
            low: The lower bound (inclusive) of the random integer range.
            high: The upper bound (inclusive) of the random integer range.

        Returns:
            A random integer value of type Int16 between low and high (inclusive).
        """
        return low + self.randint16() % (high - low + 1)

    fn randint32(self) -> Int:
        """
        Generates a random integer of type Int32.

        Returns:
            A random integer value of type Int32.
        """
        return self.lcg() % int(self.Int32MAX)

    fn randint32(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer between low and high (inclusive) of type Int32.

        Args:
            low: The lower bound (inclusive) of the random integer range.
            high: The upper bound (inclusive) of the random integer range.

        Returns:
            A random integer value of type Int32 between low and high (inclusive).
        """
        return low + self.randint32() % (high - low + 1)

    fn randint64(self) -> Int:
        """
        Generates a random integer of type Int64.

        Returns:
            A random integer value of type Int64.
        """
        return self.lcg() % int(self.Int64MAX)

    fn randint64(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer between low and high (inclusive) of type Int64.

        Args:
            low: The lower bound (inclusive) of the random integer range.
            high: The upper bound (inclusive) of the random integer range.

        Returns:
            A random integer value of type Int64 between low and high (inclusive).
        """
        return low + self.randint64() % (high - low + 1)

    fn randint(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer between low and high (inclusive).

        Args:
            low: The lower bound (inclusive) of the random integer range.
            high: The upper bound (inclusive) of the random integer range.

        Returns:
            A random integer value between low and high.
        """
        return low + self.randint64() % (high - low + 1)

    fn randf(self) -> Float32:
        """
        Generates a random floating-point number.

        Returns:
            A random floating-point number.
        """
        var random_int = self.lcg()
        var scaled_random = Float32(random_int) * 0.0000000003
        return Float32(scaled_random % self.F32Max)

    fn randf16(self, low: Float16, high: Float16) -> Float16:
        """
        Generates a random floating-point number in the range [low, high).

        Args:
            low: The lower bound (inclusive) of the random floating-point range.
            high: The upper bound (exclusive) of the random floating-point range.

        Returns:
            A random floating-point number within the specified range.
        """
        return Float16(low + self.randf16() % (high - low + 1))   

    fn randf16(self) -> Float16:
        """
        Generates a random floating-point number of type Float16.

        Returns:
            A random floating-point number of type Float16.
        """
        var random_int = self.lcg() % 65504
        var scaled_random = Float16(random_int) * 0.00001
        return Float16(scaled_random % self.F16Max)

    fn randf32(self, low: Float32, high: Float32) -> Float32:
        """
        Generates a random floating-point number in the range [low, high).

        Args:
            low: The lower bound (inclusive) of the random floating-point range.
            high: The upper bound (exclusive) of the random floating-point range.

        Returns:
            A random floating-point number within the specified range.
        """
        return Float32(low + self.randf32() % (high - low + 1))   

    fn randf32(self) -> Float32:
        """
        Generates a random floating-point number of type Float32.

        Returns:
            A random floating-point number of type Float32.
        """
        return self.randf()

    fn randf64(self, low: FloatLiteral, high: FloatLiteral) -> Float32:
        """
        Generates a random floating-point number in the range [low, high).

        Args:
            low: The lower bound (inclusive) of the random floating-point range.
            high: The upper bound (exclusive) of the random floating-point range.

        Returns:
            A random floating-point number within the specified range.
        """
        return Float64(low + self.randf64() % (high - low + 1))   

    fn randf64(self) -> Float64:
        """
        Generates a random floating-point number of type Float64.

        Returns:
            A random floating-point number of type Float64.
        """
        var random_int = self.lcg()
        var scaled_random = Float64(random_int) * 0.0000000000001
        return Float64(scaled_random % self.F64Max)