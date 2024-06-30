from net.tensor.utils import get_broadcast_index, broadcast_shapes

@always_inline("nodebug")
fn is_compatible(A: List[Int], B: List[Int]) -> Bool:
    for i in range(len(A)):
        if A[i] != B[i]:
            print(
                "Incompatible Shapes: Tensors must have the same shape got [",
                A[i],
                "] and [",
                B[i],
                "] at [",
                i,
                "]",
            )
            return False
    return True


@always_inline("nodebug")
fn tensor_op[
    dtype: DType,
    func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
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
    if not is_compatible(t1.shapes().Shapes(), t2.shapes().Shapes()):
        print(Error("Tensors must be in same shape"))
        exit(1)
    alias nelts = simdwidthof[dtype]() * 2
    var result = Tensor[dtype](t1.shapes())
    var num_elements = result.num_elements()

    @parameter
    fn operation[nelts: Int](idx: Int):
        result.store[nelts](
            idx, func[dtype, nelts](t1.load[nelts](idx), t2.load[nelts](idx))
        )

    vectorize[operation, nelts](num_elements - (num_elements % nelts))

    for i in range(num_elements - (num_elements % nelts), num_elements):
        result.store(i, func(t1.load(i), t2.load(i)))
    return result^


@always_inline("nodebug")
fn scalar_op[
    dtype: DType,
    func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](Input: Tensor[dtype], value: SIMD[dtype, 1]) -> Tensor[dtype]:
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
    constrained[dtype.is_numeric(), "the Tensor type must be numeric"]()
    alias nelts = simdwidthof[dtype]() * 2
    var Output = Tensor[dtype](Input.shapes())
    var num_elements = Output.num_elements()

    @parameter
    fn operation[nelts: Int](idx: Int):
        Output.store[nelts](idx, func(Input.load[nelts](idx), value))

    vectorize[operation, nelts, unroll_factor=4](
        num_elements - (num_elements % nelts)
    )

    for i in range(num_elements - (num_elements % nelts), num_elements):
        Output.store(i, func(Input.load(i), value))

    return Output^


@always_inline("nodebug")
fn Broadcast_op[
    dtype: DType,
    func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """Performs an element-wise operation on two tensors using broadcasting."""
    var result_shape = broadcast_shapes(tensor1.shapes(), tensor2.shapes())
    var result = Tensor[dtype](result_shape)
    var num_elements = result.num_elements()
    alias nelts = simdwidthof[dtype]() * 2

    @parameter
    fn vec_op[nelts: Int](i: Int):
        var flat_index1 = get_broadcast_index(i, tensor1.shapes(), result_shape)
        var flat_index2 = get_broadcast_index(i, tensor2.shapes(), result_shape)

        result.store[nelts](
            i,
            func[dtype, nelts](
                tensor1.load[nelts](flat_index1),
                tensor2.load[nelts](flat_index2),
            ),
        )

    vectorize[vec_op, nelts](num_elements - (num_elements % nelts))

    for i in range(num_elements - (num_elements % nelts), num_elements):
        var flat_index1 = get_broadcast_index(i, tensor1.shapes(), result_shape)
        var flat_index2 = get_broadcast_index(i, tensor2.shapes(), result_shape)
        result.store(
            i, func(tensor1.load(flat_index1), tensor2.load(flat_index2))
        )

    return result^


fn operate[
    type: DType,
    func: fn[type: DType, nelts: Int] (
        SIMD[type, nelts], SIMD[type, nelts]
    ) -> SIMD[type, nelts],
](self: Tensor[type], other: Tensor[type]) -> Tensor[type]:
    constrained[type.is_numeric(), "the Tensor type must be numeric"]()
    if is_compatible(self.shapes().Shapes(), other.shapes().Shapes()):
        return tensor_op[type, func](self, other)
    else:
        return Broadcast_op[type, func](self, other)


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
    if a.rank < 1 or b.rank < 1:
        return False

    if a.rank == 1 and b.rank == 1:
        return a[0] == b[0]

    if a.rank == 1 and b.rank > 1:
        return a[0] == b[b.rank - 2]

    if b.rank == 1 and a.rank > 1:
        return a[a.rank - 1] == b[0]

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
    var max_batch_rank = max(shape1.rank - 2, shape2.rank - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            print(
                "Error: Incompatible dimensions at index",
                i,
                ":",
                dim1,
                "vs",
                dim2,
            )
            exit(1)

        batch_dims.append(max(dim1, dim2))

    if shape1.rank > 1 and shape2.rank > 1:
        batch_dims.append(shape1[shape1.rank - 2])
        batch_dims.append(shape2[shape2.rank - 1])
    elif shape1.rank > 1 and shape2.rank == 1:
        batch_dims.append(shape1[shape1.rank - 2])
    elif shape1.rank == 1 and shape2.rank > 1:
        batch_dims.append(shape2[shape2.rank - 1])

    return shape(batch_dims)