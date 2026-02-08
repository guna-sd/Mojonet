"""
Graph: Computation graph for automatic differentiation.

This module provides infrastructure for tracking operations and
executing backward passes through the computation graph.
"""

from memory import UnsafePointer, alloc, memset_zero
from .tensor import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Graph Node Data
# ═══════════════════════════════════════════════════════════════════════════


struct GradContext(Copyable, Movable):
    """Context storing data needed for backward pass.

    Stores parent variable IDs and saved tensors for gradient computation.
    """

    var parent_ids: List[Int]
    var saved_tensors: List[Tensor]
    var op_type: Int  # Operation type enum

    fn __init__(out self):
        """Create empty context."""
        self.parent_ids = List[Int]()
        self.saved_tensors = List[Tensor]()
        self.op_type = 0

    fn __init__(out self, op_type: Int):
        """Create context with operation type."""
        self.parent_ids = List[Int]()
        self.saved_tensors = List[Tensor]()
        self.op_type = op_type

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.parent_ids = existing.parent_ids^
        self.saved_tensors = existing.saved_tensors^
        self.op_type = existing.op_type

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.parent_ids = existing.parent_ids.copy()
        self.saved_tensors = List[Tensor]()
        for i in range(len(existing.saved_tensors)):
            self.saved_tensors.append(existing.saved_tensors[i].copy())
        self.op_type = existing.op_type

    fn copy(self) -> Self:
        """Explicit copy."""
        var result = Self()
        result.parent_ids = self.parent_ids.copy()
        for i in range(len(self.saved_tensors)):
            result.saved_tensors.append(self.saved_tensors[i].copy())
        result.op_type = self.op_type
        return result^

    fn add_parent(mut self, parent_id: Int):
        """Add a parent variable ID."""
        self.parent_ids.append(parent_id)

    fn save_tensor(mut self, t: Tensor):
        """Save a tensor for backward computation."""
        self.saved_tensors.append(t.copy())


# ═══════════════════════════════════════════════════════════════════════════
# Variable Node (stored in graph)
# ═══════════════════════════════════════════════════════════════════════════


struct VariableNode(Copyable, Movable):
    """A node in the computation graph.

    Stores the variable's gradient and backward context.
    """

    var grad: Tensor  # Accumulated gradient
    var has_grad: Bool  # Whether gradient is allocated
    var requires_grad: Bool  # Whether to track gradients
    var grad_ctx: GradContext  # Context for backward pass
    var is_leaf: Bool  # True if this is a parameter/input

    fn __init__(
        out self,
        shape: List[Int],
        requires_grad: Bool = False,
        is_leaf: Bool = True,
    ):
        """Create a variable node."""
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_ctx = GradContext()

        if requires_grad:
            self.grad = Tensor(shape.copy())
            self.has_grad = True
        else:
            self.grad = Tensor()
            self.has_grad = False

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.grad = existing.grad^
        self.has_grad = existing.has_grad
        self.requires_grad = existing.requires_grad
        self.grad_ctx = existing.grad_ctx^
        self.is_leaf = existing.is_leaf

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.grad = existing.grad.copy()
        self.has_grad = existing.has_grad
        self.requires_grad = existing.requires_grad
        self.grad_ctx = existing.grad_ctx.copy()
        self.is_leaf = existing.is_leaf

    fn copy(self) -> Self:
        """Explicit copy."""
        var result = Self(List[Int](), self.requires_grad, self.is_leaf)
        result.grad = self.grad.copy()
        result.has_grad = self.has_grad
        result.grad_ctx = self.grad_ctx.copy()
        return result^

    fn zero_grad(mut self):
        """Zero the gradient buffer."""
        if self.has_grad:
            self.grad.zero_()


# ═══════════════════════════════════════════════════════════════════════════
# Operation Types (using fn constants to avoid alias deprecation)
# ═══════════════════════════════════════════════════════════════════════════


fn OP_NONE() -> Int:
    return 0


fn OP_ADD() -> Int:
    return 1


fn OP_SUB() -> Int:
    return 2


fn OP_MUL() -> Int:
    return 3


fn OP_MATMUL() -> Int:
    return 4


fn OP_MEAN() -> Int:
    return 5


fn OP_POW() -> Int:
    return 6


fn OP_SUM() -> Int:
    return 7


fn OP_TRANSPOSE() -> Int:
    return 8


fn OP_SCALAR_MUL() -> Int:
    return 9


fn OP_BROADCAST_ADD() -> Int:
    return 10


# ═══════════════════════════════════════════════════════════════════════════
# Computation Graph
# ═══════════════════════════════════════════════════════════════════════════


struct Graph(Movable):
    """Computation graph for automatic differentiation.

    Manages variable nodes and executes backward passes.
    Uses reverse creation order for topological traversal.
    """

    var nodes: List[VariableNode]
    var data: List[Tensor]  # Tensor data storage
    var next_id: Int

    fn __init__(out self):
        """Create empty graph."""
        self.nodes = List[VariableNode]()
        self.data = List[Tensor]()
        self.next_id = 0

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.nodes = existing.nodes^
        self.data = existing.data^
        self.next_id = existing.next_id

    fn register(
        mut self, t: Tensor, requires_grad: Bool = False, is_leaf: Bool = True
    ) -> Int:
        """Register a new variable in the graph.

        Returns the variable ID.
        """
        var id = self.next_id
        self.next_id += 1

        var node = VariableNode(t.shape(), requires_grad, is_leaf)
        self.nodes.append(node^)
        self.data.append(t.copy())

        return id

    fn register_with_ctx(
        mut self,
        t: Tensor,
        var ctx: GradContext,
        requires_grad: Bool = True,
    ) -> Int:
        """Register a computed variable with backward context.

        Returns the variable ID.
        """
        var id = self.next_id
        self.next_id += 1

        var node = VariableNode(t.shape(), requires_grad, is_leaf=False)
        node.grad_ctx = ctx^
        self.nodes.append(node^)
        self.data.append(t.copy())

        return id

    fn get_data(self, id: Int) -> Tensor:
        """Get tensor data for a variable."""
        return self.data[id].copy()

    fn set_data(mut self, id: Int, t: Tensor):
        """Set tensor data for a variable."""
        self.data[id] = t.copy()

    fn get_grad(self, id: Int) -> Tensor:
        """Get gradient for a variable."""
        return self.nodes[id].grad.copy()

    fn zero_all_grads(mut self):
        """Zero all gradients in the graph."""
        for i in range(len(self.nodes)):
            self.nodes[i].zero_grad()

    fn backward(mut self, output_id: Int) raises:
        """Execute backward pass from output variable.

        Traverses graph in reverse creation order and accumulates gradients.
        """
        # Initialize output gradient to 1.0
        if not self.nodes[output_id].has_grad:
            self.nodes[output_id].grad = Tensor(self.data[output_id].shape())
            self.nodes[output_id].has_grad = True
        self.nodes[output_id].grad.fill_(1.0)

        # Traverse in reverse order (reverse topological order for DAG)
        for i in range(output_id, -1, -1):
            var node = self.nodes[i].copy()

            if not node.requires_grad:
                continue

            if node.is_leaf:
                # Leaf nodes don't propagate gradients
                continue

            var op_type = node.grad_ctx.op_type
            if op_type == OP_NONE():
                continue

            # Get output gradient
            var grad_out = self.nodes[i].grad.copy()
            var ctx = node.grad_ctx.copy()

            # Compute and accumulate parent gradients based on operation type
            if op_type == OP_ADD():
                self._backward_add(grad_out, ctx)
            elif op_type == OP_SUB():
                self._backward_sub(grad_out, ctx)
            elif op_type == OP_MUL():
                self._backward_mul(grad_out, ctx)
            elif op_type == OP_MATMUL():
                self._backward_matmul(grad_out, ctx)
            elif op_type == OP_MEAN():
                self._backward_mean(grad_out, ctx)
            elif op_type == OP_POW():
                self._backward_pow(grad_out, ctx)
            elif op_type == OP_SUM():
                self._backward_sum(grad_out, ctx)
            elif op_type == OP_SCALAR_MUL():
                self._backward_scalar_mul(grad_out, ctx)
            elif op_type == OP_BROADCAST_ADD():
                self._backward_broadcast_add(grad_out, ctx)

    # ═══════════════════════════════════════════════════════════════════════
    # Backward Functions
    # ═══════════════════════════════════════════════════════════════════════

    fn _backward_add(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for add: grad_a += grad_out, grad_b += grad_out."""
        if len(ctx.parent_ids) >= 1:
            var p0 = ctx.parent_ids[0]
            if self.nodes[p0].requires_grad:
                self.nodes[p0].grad.add_(grad_out)
        if len(ctx.parent_ids) >= 2:
            var p1 = ctx.parent_ids[1]
            if self.nodes[p1].requires_grad:
                self.nodes[p1].grad.add_(grad_out)

    fn _backward_sub(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for sub: grad_a += grad_out, grad_b -= grad_out."""
        if len(ctx.parent_ids) >= 1:
            var p0 = ctx.parent_ids[0]
            if self.nodes[p0].requires_grad:
                self.nodes[p0].grad.add_(grad_out)
        if len(ctx.parent_ids) >= 2:
            var p1 = ctx.parent_ids[1]
            if self.nodes[p1].requires_grad:
                self.nodes[p1].grad.sub_(grad_out)

    fn _backward_mul(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for mul: grad_a += grad_out * b, grad_b += grad_out * a."""
        if len(ctx.saved_tensors) < 2:
            return
        var a = ctx.saved_tensors[0].copy()
        var b = ctx.saved_tensors[1].copy()

        if len(ctx.parent_ids) >= 1:
            var p0 = ctx.parent_ids[0]
            if self.nodes[p0].requires_grad:
                # grad_a = grad_out * b
                for i in range(grad_out.size()):
                    self.nodes[p0].grad[i] += grad_out[i] * b[i]

        if len(ctx.parent_ids) >= 2:
            var p1 = ctx.parent_ids[1]
            if self.nodes[p1].requires_grad:
                # grad_b = grad_out * a
                for i in range(grad_out.size()):
                    self.nodes[p1].grad[i] += grad_out[i] * a[i]

    fn _backward_matmul(mut self, grad_out: Tensor, ctx: GradContext) raises:
        """Backward for matmul: grad_a += grad_out @ b.T, grad_b += a.T @ grad_out."""
        if len(ctx.saved_tensors) < 2:
            return
        var a = ctx.saved_tensors[0].copy()
        var b = ctx.saved_tensors[1].copy()

        if len(ctx.parent_ids) >= 1:
            var p0 = ctx.parent_ids[0]
            if self.nodes[p0].requires_grad:
                # grad_a = grad_out @ b.T
                var bt = b.T()
                var grad_a = grad_out.matmul(bt)
                self.nodes[p0].grad.add_(grad_a)

        if len(ctx.parent_ids) >= 2:
            var p1 = ctx.parent_ids[1]
            if self.nodes[p1].requires_grad:
                # grad_b = a.T @ grad_out
                var at = a.T()
                var grad_b = at.matmul(grad_out)
                self.nodes[p1].grad.add_(grad_b)

    fn _backward_mean(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for mean: grad_a += grad_out / n (broadcast)."""
        if len(ctx.parent_ids) < 1:
            return
        var p0 = ctx.parent_ids[0]
        if not self.nodes[p0].requires_grad:
            return

        var n = Float32(self.nodes[p0].grad.size())
        var grad_val = grad_out[0] / n
        for i in range(self.nodes[p0].grad.size()):
            self.nodes[p0].grad[i] += grad_val

    fn _backward_sum(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for sum: grad_a += grad_out (broadcast)."""
        if len(ctx.parent_ids) < 1:
            return
        var p0 = ctx.parent_ids[0]
        if not self.nodes[p0].requires_grad:
            return

        var grad_val = grad_out[0]
        for i in range(self.nodes[p0].grad.size()):
            self.nodes[p0].grad[i] += grad_val

    fn _backward_pow(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for pow: grad_a += n * a^(n-1) * grad_out."""
        if len(ctx.saved_tensors) < 2:
            return
        var a = ctx.saved_tensors[0].copy()
        # saved_tensors[1] stores the exponent as a 1-element tensor
        var exp = ctx.saved_tensors[1][0]

        if len(ctx.parent_ids) < 1:
            return
        var p0 = ctx.parent_ids[0]
        if not self.nodes[p0].requires_grad:
            return

        for i in range(a.size()):
            var grad = exp * (a[i] ** (exp - 1.0)) * grad_out[i]
            self.nodes[p0].grad[i] += grad

    fn _backward_scalar_mul(mut self, grad_out: Tensor, ctx: GradContext):
        """Backward for scalar mul: grad_a += grad_out * scalar."""
        if len(ctx.saved_tensors) < 1:
            return
        var scalar = ctx.saved_tensors[0][0]  # Scalar stored as 1-element tensor

        if len(ctx.parent_ids) < 1:
            return
        var p0 = ctx.parent_ids[0]
        if not self.nodes[p0].requires_grad:
            return

        for i in range(grad_out.size()):
            self.nodes[p0].grad[i] += grad_out[i] * scalar

    fn _backward_broadcast_add(
        mut self, grad_out: Tensor, ctx: GradContext
    ) raises:
        """Backward for broadcast add (bias): grad_b += sum over batch."""
        if len(ctx.parent_ids) < 2:
            return

        var p0 = ctx.parent_ids[0]  # Main tensor
        var p1 = ctx.parent_ids[1]  # Bias (1D)

        if self.nodes[p0].requires_grad:
            self.nodes[p0].grad.add_(grad_out)

        if self.nodes[p1].requires_grad:
            # Sum over batch dimension
            var bias_size = self.nodes[p1].grad.size()
            var grad_shape = grad_out.shape()
            var batch = grad_shape[0]

            for j in range(bias_size):
                var sum: Float32 = 0.0
                for i in range(batch):
                    sum += grad_out[i * bias_size + j]
                self.nodes[p1].grad[j] += sum
