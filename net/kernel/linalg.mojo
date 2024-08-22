from sys.intrinsics import PrefetchOptions, prefetch

alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()


@always_inline("nodebug")
fn mm[
    T: DType
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs matrix multiplication on matrices A and B and stores the result in matrix C with vectorization and parallelization.

    Args:
        A: Pointer to the first matrix.
        B: Pointer to the second matrix.
        C: Pointer to the result matrix.
        m: Number of rows in matrix A.
        n: Number of columns in matrix A and number of rows in matrix B.
        p: Number of columns in matrix B.
    """
    alias nelts = simdwidthof[T]() * 2
    alias unroll = 4

    @parameter
    fn calc_row(i: Int):
        for k in range(n):

            @parameter
            fn dot[nelts: Int](n_idx: Int):
                prefetch[PREFETCH_READ](A + (i * n + k + nelts))
                prefetch[PREFETCH_READ](B + (k * p + n_idx + nelts))
                prefetch[PREFETCH_WRITE](C + (i * p + n_idx + nelts))

                C.store[width=nelts](
                    i * p + n_idx,
                    A.load[width=nelts](i * n + k).fma(
                        B.load[width=nelts](k * p + n_idx),
                        C.load[width=nelts](i * p + n_idx),
                    ),
                )

            vectorize[dot, nelts, unroll_factor=unroll](size=p - (p % nelts))

        for j in range(p - (p % nelts), p):
            var acc_sum = Scalar[T](0)
            for k in range(n):
                prefetch[PREFETCH_READ](A + (i * n + k + 1))
                prefetch[PREFETCH_READ](B + (k * p + j + 1))
                acc_sum = A[i * n + k].fma(B[k * p + j], acc_sum)
            prefetch[PREFETCH_WRITE](C + (i * p + j + 1))
            C[i * p + j] = acc_sum

    parallelize[calc_row](m, m)


@always_inline("nodebug")
fn fusedmm[
    T: DType, func: fn[T: DType, nelts: Int] (SIMD[T, nelts]) -> SIMD[T, nelts]
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs matrix multiplication on matrices A and B and stores the result in matrix C with vectorization and parallelization.

    Args:
        A: Pointer to the first matrix.
        B: Pointer to the second matrix.
        C: Pointer to the result matrix.
        m: Number of rows in matrix A.
        n: Number of columns in matrix A and number of rows in matrix B.
        p: Number of columns in matrix B.
    """
    alias nelts = simdwidthof[T]() * 2
    alias unroll = 4

    @parameter
    fn calc_row(i: Int):
        for k in range(n):

            @parameter
            fn dot[nelts: Int](n_idx: Int):
                prefetch[PREFETCH_READ](A + (i * n + k + nelts))
                prefetch[PREFETCH_READ](B + (k * p + n_idx + nelts))
                prefetch[PREFETCH_WRITE](C + (i * p + n_idx + nelts))
                C.store[width=nelts](
                    i * p + n_idx,
                    func[T, nelts](
                        A.load[width=nelts](i * n + k).fma(
                            B.load[width=nelts](k * p + n_idx),
                            C.load[width=nelts](i * p + n_idx),
                        ),
                    ),
                )

            vectorize[dot, nelts, unroll_factor=unroll](size=p - (p % nelts))

        for j in range(p - (p % nelts), p):
            var acc_sum = Scalar[T](0)
            for k in range(n):
                prefetch[PREFETCH_READ](A + (i * n + k + 1))
                prefetch[PREFETCH_READ](B + (k * p + j + 1))
                acc_sum = A[i * n + k].fma(B[k * p + j], acc_sum)
            prefetch[PREFETCH_WRITE](C + (i * p + j + 1))
            C[i * p + j] = func[T, 1](acc_sum)

    parallelize[calc_row](m, m)


@always_inline("nodebug")
fn fusedscalarmm[
    T: DType,
    func: fn[T: DType, nelts: Int] (SIMD[T, nelts], SIMD[T, nelts]) -> SIMD[
        T, nelts
    ],
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    Scalar_value: Scalar[T],
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs matrix multiplication on matrices A and B and stores the result in matrix C with vectorization and parallelization.

    Args:
        A: Pointer to the first matrix.
        B: Pointer to the second matrix.
        C: Pointer to the result matrix.
        Scalar_value: Scalar value to be fused.
        m: Number of rows in matrix A.
        n: Number of columns in matrix A and number of rows in matrix B.
        p: Number of columns in matrix B.
    """
    alias nelts = simdwidthof[T]() * 2
    alias unroll = 4

    @parameter
    fn calc_row(i: Int):
        for k in range(n):

            @parameter
            fn dot[nelts: Int](n_idx: Int):
                prefetch[PREFETCH_READ](A + (i * n + k + nelts))
                prefetch[PREFETCH_READ](B + (k * p + n_idx + nelts))
                prefetch[PREFETCH_WRITE](C + (i * p + n_idx + nelts))
                C.store[width=nelts](
                    i * p + n_idx,
                    func(
                        A.load[width=nelts](i * n + k).fma(
                            B.load[width=nelts](k * p + n_idx),
                            C.load[width=nelts](i * p + n_idx),
                        ),
                        Scalar_value,
                    ),
                )

            vectorize[dot, nelts, unroll_factor=unroll](size=p - (p % nelts))

        for j in range(p - (p % nelts), p):
            var acc_sum = Scalar[T](0)
            for k in range(n):
                prefetch[PREFETCH_READ](A + (i * n + k + 1))
                prefetch[PREFETCH_READ](B + (k * p + j + 1))
                acc_sum = A[i * n + k].fma(B[k * p + j], acc_sum)
            prefetch[PREFETCH_WRITE](C + (i * p + j + 1))
            C[i * p + j] = func(acc_sum, Scalar_value)

    parallelize[calc_row](m, m)


@always_inline("nodebug")
fn mm3d2d[
    T: DType
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    M: Int,
    N: Int,
    K: Int,
    Q: Int,
):
    """
    Performs matrix multiplication on a 3D tensor A and a 2D matrix B and stores the result in a 3D tensor C.
    `A = (M,N,K) B = (Q,K) -> C = (M,N,Q)`.

    Args:
        A: Pointer to the 3D tensor.
        B: Pointer to the 2D matrix.
        C: Pointer to the result 3D tensor.
        M: Size of the first dimension of tensor A.
        N: Size of the second dimension of tensor A and the number of rows in matrix B.
        K: Size of the third dimension of tensor A and number of columns in matrix B.
        Q: Number of columns in matrix A.
    """
    alias nelts = simdwidthof[T]() * 2

    @parameter
    fn _calc(b: Int):
        for t in range(N):
            var out_bt: UnsafePointer[Scalar[T]] = C + b * N * Q + t * Q
            var inp_bt: UnsafePointer[Scalar[T]] = A + b * N * K + t * K

            for o in range(Q):
                var val: SIMD[T, 1] = 0
                var wrow: UnsafePointer[Scalar[T]] = B + o * K

                @parameter
                fn _op[width: Int](iv: Int):
                    var t = inp_bt.load[width=width](iv) * wrow.load[
                        width=width
                    ](iv)
                    val += t.reduce_add()

                vectorize[_op, nelts, unroll_factor=4](size=K)

                out_bt[o] = val

    parallelize[_calc](M, M)


@always_inline("nodebug")
fn Compute_blocks[
    T: DType
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    i_outer: Int,
    i_limit: Int,
    j_outer: Int,
    j_limit: Int,
    k_outer: Int,
    k_limit: Int,
    n: Int,
    p: Int,
):
    """
    Multiply blocks of matrices A and B and accumulate the result in matrix C.

    Args:
        A: Pointer to the first matrix.
        B: Pointer to the second matrix.
        C: Pointer to the result matrix.
        i_outer: Start index for the outer loop (i).
        i_limit: End index for the outer loop (i).
        j_outer: Start index for the outer loop (j).
        j_limit: End index for the outer loop (j).
        k_outer: Start index for the outer loop (k).
        k_limit: End index for the outer loop (k).
        n: Number of columns in matrix A and number of rows in matrix B.
        p: Number of columns in matrix B.
    """
    alias nelts = simdwidthof[T]() * 2

    for i in range(i_outer, i_limit):
        for j in range(j_outer, j_limit):

            @parameter
            fn dot_product[nelts: Int](`_`: Int):
                var acc_sum = SIMD[T, nelts](0)
                for k in range(k_outer, k_limit):
                    prefetch[PREFETCH_READ](A + (i * n + k + nelts))
                    prefetch[PREFETCH_READ](B + (k * p + j + nelts))
                    acc_sum += A.load[width=nelts](i * n + k) * B.load[
                        width=nelts
                    ](k * p + j)
                prefetch[PREFETCH_WRITE](C + (i * p + j + nelts))
                C.store[width=nelts](i * p + j, acc_sum)

            vectorize[dot_product, nelts, unroll_factor=4](
                size=k_limit - k_outer
            )

        var acc_sum = Scalar[T](0)
        for k_idx in range(k_outer, k_limit):
            prefetch[PREFETCH_READ](A + (i * n + k_idx + 1))
            prefetch[PREFETCH_READ](B + (k_idx * p + j_outer + 1))
            acc_sum += A[i * n + k_idx] * B[k_idx * p + j_outer]
        prefetch[PREFETCH_WRITE](C + (i * p + j_outer + 1))
        C[i * p + j_outer] = acc_sum


# For now using this Tiled version is slower than the mm version will be optimized in future...
# TODO : write a more efficient and also device-specific optimizations for better performance.
@always_inline("nodebug")
fn tmm[
    T: DType
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    C: UnsafePointer[Scalar[T]],
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs tiled matrix multiplication on matrices A and B and stores the result in matrix C.

    Args:
        A: Pointer to the first matrix.
        B: Pointer to the second matrix.
        C: Pointer to the result matrix.
        m: Number of rows in matrix A.
        n: Number of columns in matrix A and number of rows in matrix B.
        p: Number of columns in matrix B.
    """
    alias block_size = 8

    for i_outer in range(0, m, block_size):
        for j_outer in range(0, p, block_size):
            for k_outer in range(0, n, block_size):
                var i_limit = min(i_outer + block_size, m)
                var j_limit = min(j_outer + block_size, p)
                var k_limit = min(k_outer + block_size, n)

                Compute_blocks[T](
                    A,
                    B,
                    C,
                    i_outer,
                    i_limit,
                    j_outer,
                    j_limit,
                    k_outer,
                    k_limit,
                    n,
                    p,
                )


@always_inline("nodebug")
fn bmm[
    T: DType
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    inout C: UnsafePointer[Scalar[T]],
    b: Int,
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs batch matrix multiplication on batches of matrices A and B and stores the result in matrix C.

    Args:
        A: Pointer to the first batch of matrices.
        B: Pointer to the second batch of matrices.
        C: Pointer to the result batch of matrices.
        b: Number of batches.
        m: Number of rows in each matrix of A.
        n: Number of columns in each matrix of A and number of rows in each matrix of B.
        p: Number of columns in each matrix of B.
    """

    @parameter
    fn MM(batch: Int):
        mm[T](
            A + batch * (m * n),
            B + batch * (n * p),
            C + batch * (m * p),
            m,
            n,
            p,
        )

    for batch in range(b):
        MM(batch)


fn fusedbmm[
    T: DType, func: fn[T: DType, nelts: Int] (SIMD[T, nelts]) -> SIMD[T, nelts]
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    inout C: UnsafePointer[Scalar[T]],
    b: Int,
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs batch matrix multiplication on batches of matrices A and B and stores the result in matrix C.

    Args:
        A: Pointer to the first batch of matrices.
        B: Pointer to the second batch of matrices.
        C: Pointer to the result batch of matrices.
        b: Number of batches.
        m: Number of rows in each matrix of A.
        n: Number of columns in each matrix of A and number of rows in each matrix of B.
        p: Number of columns in each matrix of B.
    """

    @parameter
    fn MM(batch: Int):
        fusedmm[T, func](
            A + batch * (m * n),
            B + batch * (n * p),
            C + batch * (m * p),
            m,
            n,
            p,
        )

    for batch in range(b):
        MM(batch)


fn fusedScalarbmm[
    T: DType,
    func: fn[T: DType, nelts: Int] (SIMD[T, nelts], SIMD[T, nelts]) -> SIMD[
        T, nelts
    ],
](
    A: UnsafePointer[Scalar[T]],
    B: UnsafePointer[Scalar[T]],
    inout C: UnsafePointer[Scalar[T]],
    Scalar_value: SIMD[T, 1],
    b: Int,
    m: Int,
    n: Int,
    p: Int,
):
    """
    Performs batch matrix multiplication on batches of matrices A and B and stores the result in matrix C.

    Args:
        A: Pointer to the first batch of matrices.
        B: Pointer to the second batch of matrices.
        C: Pointer to the result batch of matrices.
        Scalar_value: Scalar value to be fused.
        b: Number of batches.
        m: Number of rows in each matrix of A.
        n: Number of columns in each matrix of A and number of rows in each matrix of B.
        p: Number of columns in each matrix of B.
    """

    @parameter
    fn MM(batch: Int):
        fusedscalarmm[T, func](
            A + batch * (m * n),
            B + batch * (n * p),
            C + batch * (m * p),
            Scalar_value,
            m,
            n,
            p,
        )

    for batch in range(b):
        MM(batch)


@always_inline("nodebug")
fn matmul[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs matrix multiplication on two tensors and returns the resulting tensor.

    Args:
        tensor1: The first tensor in the multiplication.
        tensor2: The second tensor in the multiplication.

    Returns:
        A new tensor resulting from the batch matrix multiplication of the two input tensors.
    """
    if tensor1.rank() > 2 and tensor2.rank() > 2:
        handle_issue(
            "matrix multiplication only works with 2d use batch_matmul for"
            " tensor with rank > 2."
        )

    var result_shape = calculate_shapes(
        tensor1.tensor.shape, tensor2.tensor.shape
    )
    var result = Tensor[dtype](result_shape)

    var A = tensor1.tensor.data
    var B = tensor2.tensor.data
    var C = result.tensor.data

    var m = tensor1.tensor.shape[-2]
    var n = tensor1.tensor.shape[-1]
    var p = result.tensor.shape[-1]

    if tensor1.rank() == 3 and tensor2.rank() == 2:
        mm3d2d(
            A,
            B,
            C,
            tensor1.tensor.shape[0],
            tensor1.tensor.shape[1],
            tensor1.tensor.shape[2],
            tensor2.tensor.shape[1],
        )

    mm[dtype](A, B, C, m, n, p)
    return result^


@always_inline("nodebug")
fn batch_matmul[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs batch matrix multiplication on two tensors and returns the resulting tensor.

    Args:
        tensor1: The first tensor in the multiplication.
        tensor2: The second tensor in the multiplication.

    Returns:
        A new tensor resulting from the batch matrix multiplication of the two input tensors.
    """
    if tensor1.rank() <= 2 and tensor2.rank() <= 2:
        handle_issue("batch matrix multiplication requires a rank >= 3")

    var result_shape = calculate_shapes(tensor1.shapes(), tensor2.shapes())

    var result = Tensor[dtype](result_shape)
    var tensorA = tensor1
    var tensorB = tensor2

    var A = tensorA.tensor.data
    var B = tensorB.tensor.data
    var C = result.tensor.data

    var b = result.tensor.shape[0]
    var m = tensorA.tensor.shape[-2]
    var n = tensorA.tensor.shape[-1]
    var p = result.tensor.shape[-1]

    bmm[dtype](A, B, C, b, m, n, p)

    return result^


@always_inline("nodebug")
fn bmm[
    dtype: DType
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs batch matrix multiplication on two tensors and returns the resulting tensor.

    Args:
        tensor1: The first tensor in the multiplication.
        tensor2: The second tensor in the multiplication.

    Returns:
        A new tensor resulting from the batch matrix multiplication of the two input tensors.
    """
    if tensor1.rank() <= 2 and tensor2.rank() <= 2:
        handle_issue("batch matrix multiplication requires a rank >= 3")

    var result_shape = calculate_shapes(tensor1.shapes(), tensor2.shapes())

    var result = Tensor[dtype](result_shape)
    var tensorA = tensor1
    var tensorB = tensor2

    var A = tensorA.tensor.data
    var B = tensorB.tensor.data
    var C = result.tensor.data

    var b = result.tensor.shape[0]
    var m = tensorA.tensor.shape[-2]
    var n = tensorA.tensor.shape[-1]
    var p = result.tensor.shape[-1]

    bmm[dtype](A, B, C, b, m, n, p)

    return result^


@always_inline("nodebug")
fn fusedbmm[
    dtype: DType,
    func: fn[T: DType, nelts: Int] (SIMD[T, nelts]) -> SIMD[T, nelts],
](tensor1: Tensor[dtype], tensor2: Tensor[dtype]) -> Tensor[dtype]:
    """
    Performs batch matrix multiplication on two tensors and returns the resulting tensor.

    Args:
        tensor1: The first tensor in the multiplication.
        tensor2: The second tensor in the multiplication.

    Returns:
        A new tensor resulting from the batch matrix multiplication of the two input tensors.
    """
    if tensor1.rank() <= 2 and tensor2.rank() <= 2:
        handle_issue("batch matrix multiplication requires a rank >= 3")

    var result_shape = calculate_shapes(tensor1.shapes(), tensor2.shapes())

    var result = Tensor[dtype](result_shape)
    var tensorA = tensor1
    var tensorB = tensor2

    var A = tensorA.tensor.data
    var B = tensorB.tensor.data
    var C = result.tensor.data

    var b = result.tensor.shape[0]
    var m = tensorA.tensor.shape[-2]
    var n = tensorA.tensor.shape[-1]
    var p = result.tensor.shape[-1]

    fusedbmm[dtype, func](A, B, C, b, m, n, p)

    return result^


@always_inline("nodebug")
fn fusedScalarbmm[
    dtype: DType,
    func: fn[T: DType, nelts: Int] (SIMD[T, nelts], SIMD[T, nelts]) -> SIMD[
        T, nelts
    ],
](
    tensor1: Tensor[dtype], tensor2: Tensor[dtype], Scalar_value: SIMD[dtype, 1]
) -> Tensor[dtype]:
    """
    Performs batch matrix multiplication on two tensors and returns the resulting tensor.

    Args:
        tensor1: The first tensor in the multiplication.
        tensor2: The second tensor in the multiplication.
        Scalar_value: The scalar value to be fused.

    Returns:
        A new tensor resulting from the batch matrix multiplication of the two input tensors.
    """
    if tensor1.rank() <= 2 and tensor2.rank() <= 2:
        handle_issue("batch matrix multiplication requires a rank >= 3")

    var result_shape = calculate_shapes(tensor1.shapes(), tensor2.shapes())

    var result = Tensor[dtype](result_shape)
    var tensorA = tensor1
    var tensorB = tensor2

    var A = tensorA.tensor.data
    var B = tensorB.tensor.data
    var C = result.tensor.data

    var b = result.tensor.shape[0]
    var m = tensorA.tensor.shape[-2]
    var n = tensorA.tensor.shape[-1]
    var p = result.tensor.shape[-1]

    fusedScalarbmm[dtype, func](A, B, C, Scalar_value, b, m, n, p)

    return result^
