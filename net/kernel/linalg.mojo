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
            result.__setitem__(List[Int](i, j), sum)
        vectorize[index, nelts](n)
    parallelize[multiply_and_sum](m)
    return result

