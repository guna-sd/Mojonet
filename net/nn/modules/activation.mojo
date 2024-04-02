from algorithm import parallelize, vectorize
from tensor import Tensor
from math import max, pow, exp
from algorithm._gpu.reduction import Context



# fn relu[T : DType](Inputs : Tensor[T], Outputs : Tensor[T]):

#     alias nelts = simdwidthof[T]()

#     @parameter
#     fn calculate_row(row : Int):
#         @parameter
#         fn _relu[nelts : Int](value: Int):
#             pass

'''
struct Relu[T : DType]:
    var Inputs: Tensor[T]
    var Outputs: Tensor[T]

    fn __init__(inout self, Inputs: Tensor[T]) -> Tensor[T]:
'''

fn main():
    from builtin.hash import _hash_simd
    var vector = SIMD[DType.int64, 4](1, 2, 3, 4)
    var hash = _hash_simd(vector)
    print(vector)