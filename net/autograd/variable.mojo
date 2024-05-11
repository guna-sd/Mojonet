from utils.variant import Variant
from collections import Optional

trait operation:
    fn forward(self):
        ...
    fn backward(self):
        ...

@value
struct variable:
    var name : String
    var shape : shape
    var type : DType
    var trainable : Bool
    var tensor : Tensor
    var requires_grad: Bool
    var grad : Optional[Tensor[]]

    fn __init__(inout self : Self, name : String, shape : shape, type : DType, trainable : Bool, tensor : Tensor[], requires_grad : Bool, grad : Optional[Tensor[]]):
        self.name = name
        self.shape = shape
        self.type = type
        self.trainable = trainable
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.grad = grad
    
    fn backward(inout self : Self):
        ...

struct GradientTape:
    var operations : List[variable]
    def __init__(inout self):
        self.operations = List[variable]()

    def record_operation(self, op : variable):
        self.operations.append(op)

    def backward(self, target):
        target.grad = 1  # Initialize gradient of target to 1
        for op in range(self.operations.__len__()-1,-1,-1):
            self.operations[op].backward()  # Compute gradients for each operation
