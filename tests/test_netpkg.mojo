from net.tensor import Tensor
from net.nn import GeLU

fn main():
    var tensor = Tensor[DType.bfloat16](2,2)
    tensor.rand()
    var gelu = GeLU[DType.bfloat16]()
    print(tensor)
    var out = gelu.forward(tensor)
    print(out)