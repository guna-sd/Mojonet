from net.tensor import Tensor
from algorithm import vectorize, parallelize
from sys.info import num_physical_cores, num_logical_cores

@always_inline
fn tensor_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise operation on two tensors of equal shape."""
    var shape = t1.shape == t2.shape
    var elm = t1.num_elements() == t2.num_elements()
    if shape != elm: 
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