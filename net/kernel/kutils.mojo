@always_inline("nodebug")
fn is_compatible(A : List[Int], B : List[Int]) -> Bool:
    for i in range(len(A)):
        if A[i] != B[i]:
            print("Incompatible Shapes: Tensors must have the same shape got [",A[i],"] and [",B[i],"] at [",i,"]")
            return False
    return True


@always_inline("nodebug")
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
    if not is_compatible(t1.shape.shapes, t2.shape.shapes): 
        print(Error("Tensors must be in same shape"))
        exit(1)
    alias nelts = 1#simdwidthof[dtype]()
    var result = Tensor[dtype](t1.shape)

    @parameter
    fn operation[nelts: Int](idx: Int):
        result.store(idx, func(t1.load(idx), t2.load(idx)))
    vectorize[operation, nelts, unroll_factor=4](t1.num_elements())
    return result


@always_inline("nodebug")
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
    alias nelts = 1#simdwidthof[dtype]()
    var Output = Tensor[dtype](Input.shape)

    @parameter
    fn operation[nelts : Int](idx : Int):
        Output.store(idx,func(Input[idx], value))
    vectorize[operation, nelts, unroll_factor=4](Input.num_elements())
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


@always_inline("nodebug")
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


@always_inline("nodebug")
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
        exit(1)

    var batch_dims = List[Int]()
    var max_batch_rank = math.max(shape1.rank() - 2, shape2.rank() - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank() - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank() - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print("Error: Incompatible dimensions at index", i, ":", dim1, "vs", dim2)
            exit(1)

        batch_dims.append(math.max(dim1, dim2))

    if shape1.rank() > 1 and shape2.rank() > 1:
        batch_dims.append(shape1[shape1.rank() - 2])
        batch_dims.append(shape2[shape2.rank() - 1])
    elif shape1.rank() > 1 and shape2.rank() == 1:
        batch_dims.append(shape1[shape1.rank() - 2])
    elif shape1.rank() ==1 and shape2.rank() > 1:
        batch_dims.append(shape2[shape2.rank() - 1]) 

    return shape(batch_dims)

struct randn:
    alias F16 : Float16 = Float16.MAX_FINITE
    alias F32 : Float32 = 3.4028234481811523
    alias F64 : Float64 = 3.1415926535897931

    var _seed : Optional[Int]

    fn __init__(inout self):
        self._seed = time.now()

    fn __init__(inout self, seed: Int):
        """
        Initializes the random number generator with the provided seed.

        Args:
            seed: The seed value.
        """
        self._seed = seed

    @always_inline("nodebug")
    fn seed(inout self):
        """
        Seeds the random number generator using the current time.
        """
        self._seed = time.now()

    @always_inline("nodebug")
    fn seed(inout self, seed : Int):
        """
        Seeds the random number generator with the specified seed.

        Args:
            seed: Int The seed value.
        """
        self._seed = seed

    @always_inline("nodebug")
    fn lcg(self) -> Int:
        return (self._seed.value()[] * 1103515245 + 12345) % 65504_1234

    @staticmethod
    @always_inline("nodebug")
    fn u64(inout state : UInt64) -> UInt64:
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return ((state * 0x2545F4914F6CDD1D) >> 32).cast[DType.uint64]()

    @always_inline("nodebug")
    fn randint8(self) -> Int8:
        """
        Generates a random integer of type Int8.

        Returns:
            A random integer value of type Int8.
        """
        var val = UInt64(self._seed.value()[])
        return Int8((self.u64(val)).cast[DType.int8]()) % Int8.MAX_FINITE

    @always_inline("nodebug")
    fn randint16(self) -> Int16:
        """
        Generates a random integer of type Int16.

        Returns:
            A random integer value of type Int16.
        """
        var val = UInt64(self._seed.value()[])
        return Int16((self.u64(val)).cast[DType.int16]()) % Int16.MAX_FINITE

    @always_inline("nodebug")
    fn randint32(self) -> Int32:
        """
        Generates a random integer of type Int32.

        Returns:
            A random integer value of type Int32.
        """
        var val = UInt64(self._seed.value()[])
        return Int32((self.u64(val)).cast[DType.int32]()) % Int32.MAX_FINITE

    @always_inline("nodebug")
    fn randint64(self) -> Int64:
        """
        Generates a random integer of type Int64.

        Returns:
            A random integer value of type Int64.
        """
        var val = UInt64(self._seed.value()[])
        return Int64((self.u64(val)).cast[DType.int64]()) % Int64.MAX_FINITE

    @always_inline("nodebug")
    fn randf(self) -> Float32:
        """
        Generates a random floating-point number.

        Returns:
            A random floating-point number.
        """
        var val = UInt64(self._seed.value()[])
        return Float32((self.u64(val) >> 8).cast[DType.float32]() % self.F32)

    @always_inline("nodebug")
    fn randf16(self) -> Float16:
        """
        Generates a random floating-point number of type Float16.

        Returns:
            A random floating-point number of type Float16.
        """
        var val = UInt64(self._seed.value()[])
        return Float16((self.u64(val) >> 16).cast[DType.float16]() / self.F16)

    @always_inline("nodebug")
    fn randf32(self) -> Float32:
        """
        Generates a random floating-point number of type Float32.

        Returns:
            A random floating-point number of type Float32.
        """
        return self.randf()

    @always_inline("nodebug")
    fn randf64(self) -> Float64:
        """
        Generates a random floating-point number of type Float64.

        Returns:
            A random floating-point number of type Float64.
        """
        var val = UInt64(self._seed.value()[])
        return Float64((self.u64(val) >> 16).cast[DType.float64]() % self.F64)
    

@always_inline("nodebug")
fn rand[type : DType](ptr : DTypePointer[type], count : Int):
    alias nelts = simdwidthof[type]()

    @parameter
    fn _rand[nelts : Int](i : Int):
        @parameter
        if type.is_int8():
                ptr[i] = randn().randint8().cast[type]()

        @parameter
        if type.is_int16():
            for i in range(count):
                ptr[i] = randn().randint16().cast[type]()

        @parameter
        if type.is_int32():
            for i in range(count):
                ptr[i] = randn().randint32().cast[type]()

        @parameter
        if type.is_int64():
            for i in range(count):
                ptr[i] = randn().randint64().cast[type]()

        @parameter
        if type.is_float16():
            for i in range(count):
                ptr[i] = randn().randf16().cast[type]()

        @parameter
        if type.is_float32():
            for i in range(count):
                ptr[i] = randn().randf32().cast[type]()

        @parameter
        if type.is_float64():
            for i in range(count):
                ptr[i] = randn().randf64().cast[type]()

        @parameter
        if type.is_bfloat16():
            for i in range(count):
                ptr[i] = randn().randf16().cast[type]()
    vectorize[_rand, nelts](count)
