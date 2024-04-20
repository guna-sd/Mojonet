from net.tensor import Tensor, shape
import net
from net.nn.activation import Fuctional as F


    

fn main():
    var tensor1 = Tensor[DType.bfloat16](2,4)
    var tensor2 = Tensor[DType.bfloat16](4,2)
    var tensor3 = net.ones[DType.bfloat16](2,4)
    tensor1.rand()
    tensor2.rand()
    print(tensor1)
    print(tensor2)
    print(tensor1 @ tensor2)
    print(tensor3)