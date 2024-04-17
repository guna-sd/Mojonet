from net.tensor import Tensor
from net.nn.activation import Fuctional as F


    

fn main():
    var tensor = Tensor[DType.bfloat16](2,2)
    tensor.rand()
    var fuctional = F[DType.bfloat16]()
    print(fuctional.GeLU(tensor))