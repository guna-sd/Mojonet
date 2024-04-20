from net.tensor import Tensor, shape
import net
from net.nn.activation import Fuctional as F
import math

    

fn main():
    var tensor1 = Tensor[DType.int16](2,4)
    var nshape = shape(List[Int](1,4))
    var tensor2 = Tensor[DType.int16](nshape)
    var tensor3 = net.ones[DType.bfloat16](2,4)
    tensor1.rand()
    tensor2.rand()
    print(tensor1)
    print(tensor2)
    print(tensor1.add(tensor2))
    
