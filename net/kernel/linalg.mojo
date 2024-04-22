@always_inline
fn matmul[dtype : DType](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    """Matrix multiplication of two tensors A and B 2D.
    A should be of shape (m, k) and B should be of shape (k, n).
    The result will be a tensor of shape (m, n).
    """
    var m = A.shape[0]  
    var k = A.shape[1]  
    var n = B.shape[1] 
    alias nelts = simdwidthof[dtype]()

    if k != B.shape[0]:
        print("Incompatible shapes for matrix multiplication: A.shape[1] must equal B.shape[0]")
        return A
    
    var result = Tensor[dtype](List[Int](m, n))

    @parameter
    fn multiply_and_sum(i: Int):
        @parameter
        fn index[nelts : Int](j: Int):
            var sum: SIMD[dtype,1] = 0
            for p in range(k):
                sum += A[i, p] * B[p, j]
            result[List[Int](i, j)] = sum
        vectorize[index, nelts](n)
    parallelize[multiply_and_sum](m)
    return result

fn mat_mul[type : DType](self: Tensor[type], other: Tensor[type]) -> Tensor[type]:
    if self.shape.rank() == 1 and other.shape.rank() == 1:
        return Tensor[type]()

    if not check_matmul(self.shape, other.shape):
        print("Error: Dimensions don't conform for a matmul.")
        return Tensor[type]()

    var res_dims = List[Int]()
    for i in range(self.shape.rank() - 1):
        res_dims.append(self.shape[i])
    res_dims.append(other.shape[other.shape.rank() - 1])

    var result = Tensor[type](res_dims)

    for i in range(self.num_elements()):
        for j in range(other.shape[other.shape.rank() - 1]):
            for k in range(self.shape[self.shape.rank() - 1]):
                var self_index = i * self.shape[self.shape.rank() - 1] + k
                var other_index = (
                    (i // result.shape[result.shape.rank() - 2])
                    * self.shape[self.shape.rank() - 1]
                    * result.shape[result.shape.rank() - 1]
                    + j * result.shape[result.shape.rank() - 1]
                    + k
                )
                var result_value = self.load[1](self_index) * other.load[1](other_index)
                var result_index = i * result.shape[result.shape.rank() - 1] + j
                result.store[1](result_index, result_value)

    return result



fn accumulate[dtype : DType](acc : SIMD[dtype,1], A : SIMD[dtype,1], B : SIMD[dtype,1]) -> SIMD[dtype,1]:
    
    return A.fma(B,acc)

fn matmul_kernel[type : DType](inout A: Tensor[type], inout B: Tensor[type]) -> Tensor[type]:
    """
    Matrix multiplication kernel.

    Args:
        A: The first tensor.
        B: The second tensor.

    Returns:
        The result of the matrix multiplication operation.
    """
    var res_shape = calculate_shapes(A.shape, B.shape)
    print(res_shape)

    var result = Tensor[type](res_shape)

    for i in range(0,result.shape[0]):
        for j in range(0,result.shape[1]):
            var acc: SIMD[type, 1] = 0
            for k in range(0,result.shape[-1]):
                acc = accumulate(acc, A[i, k] , B[k, j])
            result[List[Int](i, j)] = acc

    return result
