@always_inline("nodebug")
fn is_compatible(A : List[Int], B : List[Int]) -> Bool:
    for i in range(len(A)):
        if A[i] != B[i]:
            print("Incompatible Shapes: Tensors must have the same shape got [",A[i],"] and [",B[i],"] at [",i,"]")
            return False
    return True


@always_inline("nodebug")
fn tensor_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
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
    if not is_compatible(t1.shapes().shapes, t2.shapes().shapes): 
        print(Error("Tensors must be in same shape"))
        exit(1)
    alias nelts = simdwidthof[dtype]() * 2
    var result = Tensor[dtype](t1.shapes())

    @parameter
    fn operation[nelts: Int](idx: Int):
        result.store(idx, func(t1.load(idx), t2.load(idx)))
    vectorize[operation, nelts, unroll_factor=4](t1.num_elements())

    for i in range(result.num_elements() - (result.num_elements() % nelts)):
        if i >= result.num_elements():
            break
        if i % nelts == 0:
            continue
        result.store(i, func(t1.load(i), t2.load(i)))
    
    return result


@always_inline("nodebug")
fn scalar_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
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
    alias nelts = simdwidthof[dtype]() * 2
    var Output = Tensor[dtype](Input.shapes())

    @parameter
    fn operation[nelts : Int](idx : Int):
        Output.store(idx,func(Input[idx], value))
    vectorize[operation, nelts, unroll_factor=4](Input.num_elements())

    for i in range(Output.num_elements() - (Output.num_elements() % nelts)):
        if i >= Output.num_elements():
            break
        if i % nelts == 0:
            continue
        Output.store(i, func(Input.load(i), value))
    
    return Output


@always_inline("nodebug")
fn Broadcast_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]) -> SIMD[dtype, nelts]](
            tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """Performs an element-wise operation on two tensors using broadcasting."""
    var result_shape = broadcast_shapes(tensor1.shapes(), tensor2.shapes())
    var result = Tensor[dtype](result_shape)
    alias nelts = simdwidthof[dtype]() * 2

    @parameter
    fn vec_op[nelts: Int](i: Int):
        var flat_index1 = get_broadcast_index(i,tensor1.shapes(), result_shape)
        var flat_index2 = get_broadcast_index(i,tensor2.shapes(), result_shape)
        
        result.store[nelts](i,func[dtype, nelts](tensor1.load[nelts](flat_index1), tensor2.load[nelts](flat_index2)),)
    vectorize[vec_op, nelts](result.num_elements())

    for i in range(result.num_elements() - (result.num_elements() % nelts)):
        if i >= result.num_elements():
            break
        if i % nelts == 0:
            continue
        var flat_index1 = get_broadcast_index(i, tensor1.shapes(), result_shape)
        var flat_index2 = get_broadcast_index(i, tensor2.shapes(), result_shape)
        result.store(i, func(tensor1.load(flat_index1), tensor2.load(flat_index2)))
    
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
    var max_batch_rank = max(shape1.rank() - 2, shape2.rank() - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank() - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank() - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print("Error: Incompatible dimensions at index", i, ":", dim1, "vs", dim2)
            exit(1)

        batch_dims.append(max(dim1, dim2))

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

    var _seed : Int

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
        return (self._seed * 1103515245 + 12345) % 65504_1234

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
        var val = UInt64(self._seed)
        return Int8((self.u64(val) >> 2).cast[DType.int8]()) % Int8.MAX_FINITE

    @always_inline("nodebug")
    fn randint16(self) -> Int16:
        """
        Generates a random integer of type Int16.

        Returns:
            A random integer value of type Int16.
        """
        var val = UInt64(self._seed)
        return Int16((self.u64(val) >> 4).cast[DType.int16]()) % Int16.MAX_FINITE

    @always_inline("nodebug")
    fn randint32(self) -> Int32:
        """
        Generates a random integer of type Int32.

        Returns:
            A random integer value of type Int32.
        """
        var val = UInt64(self._seed)
        return Int32((self.u64(val) >> 8).cast[DType.int32]()) % Int32.MAX_FINITE

    @always_inline("nodebug")
    fn randint64(self) -> Int64:
        """
        Generates a random integer of type Int64.

        Returns:
            A random integer value of type Int64.
        """
        var val = UInt64(self._seed)
        return Int64((self.u64(val) >> 16).cast[DType.int64]()) % Int64.MAX_FINITE

    @always_inline("nodebug")
    fn randf(self) -> Float32:
        """
        Generates a random floating-point number.

        Returns:
            A random floating-point number.
        """
        var val = UInt64(self._seed)
        return Float32((self.u64(val) >> 8).cast[DType.float32]() % self.F32)

    @always_inline("nodebug")
    fn randf16(self) -> Float16:
        """
        Generates a random floating-point number of type Float16.

        Returns:
            A random floating-point number of type Float16.
        """
        var val = UInt64(self._seed)
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
        var val = UInt64(self._seed)
        return Float64((self.u64(val) >> 16).cast[DType.float64]() % self.F64)
    

@always_inline("nodebug")
fn rand[type : DType](ptr : DTypePointer[type], count : Int):

    @parameter
    fn _rand(i : Int):
        if type.is_int8():
                ptr[i] = randn().randint8().cast[type]()

        if type.is_int16():
                ptr[i] = randn().randint16().cast[type]()

        if type.is_int32():
                ptr[i] = randn().randint32().cast[type]()

        if type.is_int64():
                ptr[i] = randn().randint64().cast[type]()

        if type.is_float16():
                ptr[i] = randn().randf16().cast[type]()

        if type.is_float32():
                ptr[i] = randn().randf32().cast[type]()

        if type.is_float64():
                ptr[i] = randn().randf64().cast[type]()

        if type.is_bfloat16():
                ptr[i] = randn().randf16().cast[type]()
    parallelize[_rand](count, count)