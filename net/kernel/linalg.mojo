from algorithm import Static2DTileUnitFunc as Tile2DFunc

fn matmuld[type : DType](A: Tensor[type], B: Tensor[type]) -> Tensor[type]:
    alias nelts = simdwidthof[type]()
    var C : Tensor[type] = Tensor[type](calculate_shapes(A.shape, B.shape))
    var m = A.shape[0]  
    var k = A.shape[1]  
    var n = B.shape[1]
    
    @parameter
    fn calc_row(m_idx: Int):
        for n_idx in range(n):
            var sum: Scalar[type] = Scalar[type]()
            for k_idx in range(k):
                sum += A.load(m_idx, k_idx) * B.load(k_idx, n_idx)
            C.store(m_idx, n_idx, val=sum)
    
    parallelize[calc_row](m, m)
    return C


@always_inline
fn matmul2d[dtype : DType](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    """Matrix multiplication of two tensors A and B 2D.
    A should be of shape (m, k) and B should be of shape (k, n).
    The result will be a tensor of shape (m, n).
    """
    var m = A.shape[0]
    var k = A.shape[1]
    var n = B.shape[1]
    
    var result = Tensor[dtype](m, n)

    for i in range(m):
        for j in range(n):
            var sum: Scalar[dtype] = 0
            for p in range(k):
                sum += A[i, p] * B[p, j]
            result.store(i, j,val=sum)
    return result

# fn matmul_tiled_unrolled_parallelized[type : DType](A: Tensor[type], B: Tensor[type]) -> Tensor[type]:
#     alias nelts = simdwidthof[type]()
#     var C : Tensor[type] = Tensor[type](calculate_shapes(A.shape, B.shape))
#     var m = A.shape[0]  
#     var k = A.shape[1]  
#     var n = B.shape[1]
#     @parameter
#     fn calc_row(m: Int):
#         @parameter
#         fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
#             for k in range(y, y + tile_y):
#                 @parameter
#                 fn dot[nelts: Int](n: Int):
#                     C.store[nelts](m, n + x, val=accumulate[type, nelts](C.load[nelts](m, n + x) , A[m, k] , B.load[nelts](k, n + x)))

#                 alias unroll_factor = tile_x // nelts
#                 vectorize[dot, nelts, size=tile_x, unroll_factor=unroll_factor]()

#         alias tile_size = 4
#         tile[calc_tile, nelts * tile_size, tile_size](A.shape[1], C.shape[1])

#     parallelize[calc_row](C.shape[0], C.shape[0])
#     return C


# fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
#     for y in range(0, end_y, tile_y):
#         for x in range(0, end_x, tile_x):
#             tiled_fn[tile_x, tile_y](x, y)


fn add_list(a : List[Int], b : List[Int]) -> List[Int]:
    var temp = a
    temp.extend(b)
    return temp


@always_inline
fn matmul[dtype : DType](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    """
    Multi-dimensional matrix multiplication of two tensors A and B.
    The matrix multiplication will be applied to the last two dimensions of A and B.
    The earlier dimensions will be broadcasted accordingly.

    Parameters:
        dtype : DType.

    Args:
        A : Tensor[dtype].
        B : Tensor[dtype].

    Eg:
        A with shape (a1, ..., an, m, k) and B with shape (b1, ..., bn, k, n) will result in a tensor with shape (max(a1, b1), ..., max(an, bn), m, n).
    """

    var shapes = calculate_shapes(A.shape, B.shape)
    var result = Tensor[dtype](shapes)

    var m = shapes[-2]
    var n = shapes[-1]
    var k = A.shape[-1]

    var batch_dims : List[Int] = List[Int]()
    for i in range(len(shapes) - 2):
        batch_dims.append(shapes[i])
    var batch_shape = shape(batch_dims)

    alias nelts = simdwidthof[dtype]()
    var cores = num_physical_cores()-2 if num_physical_cores() > 4 else 2

    @parameter
    fn multiply_and_sum(batch_indices: List[Int], i: Int, j: Int):
        var sum: SIMD[dtype, 1] = 0
        
        for p in range(k):
            # A[i, p] * B[p, j] for each batch, i, j. Need to handle batch indexing.
            var a_index = add_list(batch_indices , List[Int](i, p))
            var b_index = add_list(batch_indices , List[Int](p, j))
            sum = A[a_index].fma(multiplier = B[b_index], accumulator=sum) 
        
        var result_index = add_list(batch_indices , List[Int](i, j))
        result.store(result_index, sum)

    @parameter
    fn process_batch(batch_indices: List[Int]):
        @parameter
        fn process_row(i: Int):
            @parameter
            fn process_column[nelts: Int](j : Int):
                multiply_and_sum(batch_indices, i, j)
            vectorize[process_column, nelts,unroll_factor=4](n)
        
        parallelize[process_row](m, cores)


    var total_batches = batch_shape.num_elements
    for batch_index in range(total_batches):
        var current_batch = batch_shape.indices(batch_index)
        process_batch(current_batch)

    return result


fn accumulate[dtype : DType, nelts : Int](acc : SIMD[dtype,nelts], A : SIMD[dtype,nelts], B : SIMD[dtype,nelts]) -> SIMD[dtype,nelts]:
    
    return A.fma(B,acc)


fn matmul_submatrix[type : DType](inout a: Tensor[type], inout b: Tensor[type], inout c: Tensor[type],
                    lo_m: Int, hi_m: Int, lo_n: Int, hi_n: Int, lo_k: Int, hi_k: Int):
  """
  Performs matrix multiplication on sub-matrices of the provided tensors.

  Args:
      a: The first input tensor.
      b: The second input tensor.
      c: The output tensor.
      lo_m: Starting row index for the sub-matrix in a (inclusive).
      hi_m: Ending row index for the sub-matrix in a (exclusive).
      lo_n: Starting column index for the sub-matrix in b (inclusive).
      hi_n: Ending column index for the sub-matrix in b (exclusive).
      lo_k: Starting column index for the sub-matrix in a (and row index for sub-matrix in b) (inclusive).
      hi_k: Ending column index for the sub-matrix in a (and row index for sub-matrix in b) (exclusive).
  """
  for m in range(lo_m, hi_m):
    for k in range(lo_k, hi_k):
      for n in range(lo_n, hi_n):
        c.store(m, n, val=accumulate(c.load(m, n), a.load(m, k) , b.load(k, n)))


fn matmulImplTiling[type : DType](
  left: Tensor[type],
  right: Tensor[type],
  result: Tensor[type],
  tile_size: Int,
  rows: Int,
  cols: Int
):

  var num_tiles = (cols + tile_size - 1) / tile_size

  for tile_index in range(num_tiles):
    var tile_start = tile_index * tile_size;
    var tile_end = math.min((tile_index + 1) * tile_size, cols)

    for row in range(rows):
      for inner in range(tile_start, tile_end):
        for col in range(cols):
          result.store(row, col, val=accumulate[type](
            result[row, col],
            left[row, inner],
            right[inner, col]
          ))
