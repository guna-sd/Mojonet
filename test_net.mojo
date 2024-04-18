from net.tensor import Tensor
from net.nn.activation import Fuctional as F


    

fn main():
    var tensor = Tensor[DType.bfloat16](2,4)
    tensor.rand()
    print(tensor)
    var ten = tensor.broadcast_to(List[Int](3,5))
    print(ten)