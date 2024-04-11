from .modules import *
from .modules.activation import *

@value
struct Fuctional[type : DType]:

    @staticmethod
    fn Gelu[type : DType,cores : Int=4,alg : String = "vectorize"](Input : Tensor[type]) -> Tensor[type]:
        return gelu[type,cores,alg](Input)
    fn Relu[type : DType](inout self, Input : Tensor[type]) -> Tensor[type]:
        return relu[type](Input)
    fn Sigmoid[type : DType](inout self, Input : Tensor[type]) -> Tensor[type]:
        return sigmoid[type](Input)
    fn Silu[type : DType](inout self, Input : Tensor[type]) -> Tensor[type]:
        return silu[type](Input)
    fn TanH[type : DType](inout self, Input : Tensor[type]) -> Tensor[type]:
        return tanh[type](Input)
