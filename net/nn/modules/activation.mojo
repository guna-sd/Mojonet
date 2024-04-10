from algorithm import parallelize, vectorize
from net import Tensor
from math import max, pow, exp
from algorithm._gpu.reduction import Context


@always_inline
fn relu[T: DType](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    var num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        var dt: SIMD[T, 1] = x[size]
        x[size] = dt if dt > 0 else 0.0

    parallelize[_row](num_elements, number_of_cores)


@always_inline
fn silu[T: DType](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    var num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        var dt: SIMD[T, 1] = x[size]
        x[size] = dt * (1.0 / (1.0 + math.exp(-dt)))

    parallelize[_row](num_elements, number_of_cores)

@always_inline
fn sigmoid[T: DType](inout x: Tensor[T], number_of_cores: Int = 1, size: Int = 0) -> None:
    var num_elements: Int = x.num_elements() if size == 0 else size

    @parameter
    fn _row(size: Int):
        x[size] = 1.0 / (1.0 + math.exp(-x[size]))

    parallelize[_row](num_elements, number_of_cores)


fn relu[T : DType](Inputs : Tensor[T], Outputs : Tensor[T]):

    alias nelts = simdwidthof[T]()

    @parameter
    fn _row(row : Int):
        @parameter
        fn _relu[nelts : Int](value: Int):
            pass

# @always_inline
# fn tensor_relu[T: DType, nelts: Int](inout B: Tensor[T], A: Tensor[T]):
#     @parameter
#     fn v_relu[nelts: Int](i: Int):
#         var zeros = SIMD[T, nelts]()
#         B.simd_store[nelts](
#             i,
#             (A.simd_load[nelts](i) > zeros).cast[T]() * A.simd_load[nelts](i),
#         )

#     vectorize[nelts, v_relu](B.num_elements())

    
struct Functional:
    ...



fn main():
    from builtin.hash import _hash_simd
    var vector = SIMD[DType.int64, 4](1, 2, 3, 4)
    var hash = _hash_simd(vector)
    print(vector)