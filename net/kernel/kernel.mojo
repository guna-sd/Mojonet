@value
struct Operation:
    """
    Represents an operation performed on tensors.
    """
    var name: String
    var forward: Function
    var backward: Function
    ...

@value
struct Function:
    ...

@value
struct Node:
    """
    Node in the computation graph.
    """
    var operation: Operation
    var inputs: List[Node]
    var outputs: List[Node]
    var grad_fn: Function
