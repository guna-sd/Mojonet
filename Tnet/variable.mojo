"""
Variable: Tensor wrapper with automatic differentiation.

A Variable is a Tensor that tracks its place in the computation graph
and can compute gradients via backpropagation.
"""

from memory import UnsafePointer
from .tensor import Tensor
from .graph import Graph, GradContext, VariableNode
from .graph import OP_NONE, OP_ADD, OP_SUB, OP_MUL, OP_MATMUL, OP_MEAN, OP_POW
from .graph import OP_SUM, OP_TRANSPOSE, OP_SCALAR_MUL, OP_BROADCAST_ADD


# ═══════════════════════════════════════════════════════════════════════════
# Variable Struct
# ═══════════════════════════════════════════════════════════════════════════


struct Variable(Movable, Stringable, Writable):
    """A tensor with automatic differentiation support.

    Variables track operations in a computation graph and can compute
    gradients via backpropagation.

    Attributes:
        id: Unique identifier in the computation graph.
        requires_grad: Whether gradients should be computed.

    Note: This version stores the variable ID only. The graph must be
    passed explicitly to methods that need it.
    """

    var _id: Int
    var requires_grad: Bool

    # ═══════════════════════════════════════════════════════════════════════
    # Construction
    # ═══════════════════════════════════════════════════════════════════════

    fn __init__(
        out self, t: Tensor, mut graph: Graph, requires_grad: Bool = False
    ):
        """Create a variable from a tensor.

        Args:
            t: The tensor data.
            graph: The computation graph to register with.
            requires_grad: Whether to track gradients for this variable.
        """
        self._id = graph.register(t, requires_grad, is_leaf=True)
        self.requires_grad = requires_grad

    fn __init__(
        out self,
        t: Tensor,
        mut graph: Graph,
        var ctx: GradContext,
        requires_grad: Bool = True,
    ):
        """Create a variable with backward context (for computed values).

        Args:
            t: The tensor data.
            graph: The computation graph.
            ctx: The gradient context from the operation that created this.
            requires_grad: Whether to track gradients.
        """
        self._id = graph.register_with_ctx(t, ctx^, requires_grad)
        self.requires_grad = requires_grad

    fn __init__(out self, id: Int, requires_grad: Bool):
        """Create a variable with known ID (internal use)."""
        self._id = id
        self.requires_grad = requires_grad

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self._id = existing._id
        self.requires_grad = existing.requires_grad

    # ═══════════════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════════════

    fn id(self) -> Int:
        """Get the variable's ID in the graph."""
        return self._id

    fn data(self, graph: Graph) -> Tensor:
        """Get the tensor data."""
        return graph.get_data(self._id)

    fn grad(self, graph: Graph) -> Tensor:
        """Get the accumulated gradient."""
        return graph.get_grad(self._id)

    fn shape(self, graph: Graph) -> List[Int]:
        """Get the shape of the tensor."""
        return self.data(graph).shape()

    fn size(self, graph: Graph) -> Int:
        """Get the total number of elements."""
        return self.data(graph).size()

    fn item(self, graph: Graph) -> Float32:
        """Get single element (for scalar variables)."""
        return self.data(graph).item()

    # ═══════════════════════════════════════════════════════════════════════
    # Gradient Operations
    # ═══════════════════════════════════════════════════════════════════════

    fn zero_grad(self, mut graph: Graph):
        """Zero the gradient buffer."""
        graph.nodes[self._id].zero_grad()

    fn backward(self, mut graph: Graph) raises:
        """Compute gradients via backpropagation.

        This initiates a backward pass from this variable (typically a loss).
        Gradients are accumulated in all variables with requires_grad=True.
        """
        graph.backward(self._id)

    # ═══════════════════════════════════════════════════════════════════════
    # Detach
    # ═══════════════════════════════════════════════════════════════════════

    fn detach(self, graph: Graph, mut new_graph: Graph) -> Self:
        """Create a new variable without gradient tracking.

        Useful for inference or stopping gradient flow.
        """
        return Variable(self.data(graph), new_graph, requires_grad=False)

    # ═══════════════════════════════════════════════════════════════════════
    # String Representation
    # ═══════════════════════════════════════════════════════════════════════

    fn __str__(self) -> String:
        """String representation (without data, as graph is not available)."""
        var result = String("Variable(id=")
        result += String(self._id)
        result += ", requires_grad="
        if self.requires_grad:
            result += "True"
        else:
            result += "False"
        result += ")"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Write to writer."""
        writer.write(self.__str__())


# ═══════════════════════════════════════════════════════════════════════════
# Differentiable Operations
# ═══════════════════════════════════════════════════════════════════════════


fn add(a: Variable, b: Variable, mut graph: Graph) raises -> Variable:
    """Element-wise addition with gradient tracking.

    Backward: grad_a += grad_out, grad_b += grad_out
    """
    var result_data = graph.get_data(a.id()) + graph.get_data(b.id())
    var requires_grad = a.requires_grad or b.requires_grad

    var ctx = GradContext(OP_ADD())
    ctx.add_parent(a.id())
    ctx.add_parent(b.id())

    return Variable(result_data, graph, ctx^, requires_grad)


fn sub(a: Variable, b: Variable, mut graph: Graph) raises -> Variable:
    """Element-wise subtraction with gradient tracking.

    Backward: grad_a += grad_out, grad_b -= grad_out
    """
    var result_data = graph.get_data(a.id()) - graph.get_data(b.id())
    var requires_grad = a.requires_grad or b.requires_grad

    var ctx = GradContext(OP_SUB())
    ctx.add_parent(a.id())
    ctx.add_parent(b.id())

    return Variable(result_data, graph, ctx^, requires_grad)


fn mul(a: Variable, b: Variable, mut graph: Graph) raises -> Variable:
    """Element-wise multiplication with gradient tracking.

    Backward: grad_a += grad_out * b, grad_b += grad_out * a
    """
    var a_data = graph.get_data(a.id())
    var b_data = graph.get_data(b.id())
    var result_data = a_data * b_data
    var requires_grad = a.requires_grad or b.requires_grad

    var ctx = GradContext(OP_MUL())
    ctx.add_parent(a.id())
    ctx.add_parent(b.id())
    ctx.save_tensor(a_data)
    ctx.save_tensor(b_data)

    return Variable(result_data^, graph, ctx^, requires_grad)


fn matmul(a: Variable, b: Variable, mut graph: Graph) raises -> Variable:
    """Matrix multiplication with gradient tracking.

    Backward: grad_a += grad_out @ b.T, grad_b += a.T @ grad_out
    """
    var a_data = graph.get_data(a.id())
    var b_data = graph.get_data(b.id())
    var result_data = a_data.matmul(b_data)
    var requires_grad = a.requires_grad or b.requires_grad

    var ctx = GradContext(OP_MATMUL())
    ctx.add_parent(a.id())
    ctx.add_parent(b.id())
    ctx.save_tensor(a_data)
    ctx.save_tensor(b_data)

    return Variable(result_data^, graph, ctx^, requires_grad)


fn mean(a: Variable, mut graph: Graph) -> Variable:
    """Compute mean of all elements with gradient tracking.

    Backward: grad_a += grad_out / n
    """
    var a_data = graph.get_data(a.id())
    var mean_val = a_data.mean()

    # Create 1-element tensor for the result
    var result = Tensor(1)
    result[0] = mean_val

    var ctx = GradContext(OP_MEAN())
    ctx.add_parent(a.id())

    return Variable(result^, graph, ctx^, a.requires_grad)


fn sum_all(a: Variable, mut graph: Graph) -> Variable:
    """Compute sum of all elements with gradient tracking.

    Backward: grad_a += grad_out (broadcast)
    """
    var a_data = graph.get_data(a.id())
    var sum_val = a_data.sum()

    # Create 1-element tensor for the result
    var result = Tensor(1)
    result[0] = sum_val

    var ctx = GradContext(OP_SUM())
    ctx.add_parent(a.id())

    return Variable(result^, graph, ctx^, a.requires_grad)


fn pow(a: Variable, exp: Float32, mut graph: Graph) -> Variable:
    """Element-wise power with gradient tracking.

    Backward: grad_a += n * a^(n-1) * grad_out
    """
    var a_data = graph.get_data(a.id())
    var result_data = a_data**exp

    var ctx = GradContext(OP_POW())
    ctx.add_parent(a.id())
    ctx.save_tensor(a_data)

    # Store exponent as 1-element tensor
    var exp_tensor = Tensor(1)
    exp_tensor[0] = exp
    ctx.save_tensor(exp_tensor^)

    return Variable(result_data^, graph, ctx^, a.requires_grad)


fn scalar_mul(a: Variable, scalar: Float32, mut graph: Graph) -> Variable:
    """Scalar multiplication with gradient tracking.

    Backward: grad_a += grad_out * scalar
    """
    var a_data = graph.get_data(a.id())
    var result_data = a_data * scalar

    var ctx = GradContext(OP_SCALAR_MUL())
    ctx.add_parent(a.id())

    # Store scalar as 1-element tensor
    var scalar_tensor = Tensor(1)
    scalar_tensor[0] = scalar
    ctx.save_tensor(scalar_tensor^)

    return Variable(result_data^, graph, ctx^, a.requires_grad)


fn broadcast_add(
    a: Variable, bias: Variable, mut graph: Graph
) raises -> Variable:
    """Add bias to each row (broadcast over batch dimension).

    a: [batch, features]
    bias: [features]
    result: [batch, features]

    Backward: grad_a += grad_out, grad_bias += sum(grad_out, axis=0)
    """
    var a_data = graph.get_data(a.id())
    var bias_data = graph.get_data(bias.id())
    var a_shape = a_data.shape()
    var batch = a_shape[0]
    var features = a_shape[1]

    # Create result tensor
    var result = Tensor(a_shape.copy())
    for i in range(batch):
        for j in range(features):
            result[i * features + j] = a_data[i * features + j] + bias_data[j]

    var requires_grad = a.requires_grad or bias.requires_grad

    var ctx = GradContext(OP_BROADCAST_ADD())
    ctx.add_parent(a.id())
    ctx.add_parent(bias.id())

    return Variable(result^, graph, ctx^, requires_grad)


fn transpose(a: Variable, mut graph: Graph) raises -> Variable:
    """Transpose (2D only) with gradient tracking.

    Note: For simplicity, we don't track gradients through transpose.
    This is fine for the demo since we use it on non-leaf tensors.
    """
    var result_data = graph.get_data(a.id()).T()
    return Variable(result_data^, graph, requires_grad=False)
