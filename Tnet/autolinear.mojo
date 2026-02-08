"""
AutoLinear: Linear layer using automatic differentiation.

This module provides a Linear layer that uses the autograd system
instead of manual backward computation.
"""

from .tensor import Tensor, random
from .graph import Graph
from .variable import Variable, matmul, broadcast_add


struct AutoLinear(Movable):
    """Fully connected layer using autograd: y = x @ W.T + b.

    Unlike the manual Linear layer, this uses the computation graph
    for automatic gradient computation.
    """

    var weight_id: Int  # ID of weight Variable in graph
    var bias_id: Int  # ID of bias Variable in graph
    var in_features: Int
    var out_features: Int
    var has_bias: Bool

    fn __init__(
        out self,
        in_features: Int,
        out_features: Int,
        mut graph: Graph,
        has_bias: Bool = True,
    ):
        """Create linear layer with parameters registered in graph.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            graph: The computation graph to register parameters with.
            has_bias: Whether to include a bias term.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # Initialize weight with small random values (Xavier-like)
        var weight_data = random(out_features, in_features)
        var scale = Float32(1.0) / Float32(in_features)
        for i in range(weight_data.size()):
            weight_data[i] = (weight_data[i] - 0.5) * scale

        var weight = Variable(weight_data^, graph, requires_grad=True)
        self.weight_id = weight.id()

        # Initialize bias to zero
        var bias_data = Tensor(out_features)
        var bias = Variable(bias_data^, graph, requires_grad=True)
        self.bias_id = bias.id()

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.weight_id = existing.weight_id
        self.bias_id = existing.bias_id
        self.in_features = existing.in_features
        self.out_features = existing.out_features
        self.has_bias = existing.has_bias

    fn weight(self) -> Variable:
        """Get weight variable."""
        return Variable(self.weight_id, requires_grad=True)

    fn bias(self) -> Variable:
        """Get bias variable."""
        return Variable(self.bias_id, requires_grad=True)

    fn get_weight_data(self, graph: Graph) -> Tensor:
        """Get current weight tensor."""
        return graph.get_data(self.weight_id)

    fn get_bias_data(self, graph: Graph) -> Tensor:
        """Get current bias tensor."""
        return graph.get_data(self.bias_id)

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        var count = self.in_features * self.out_features
        if self.has_bias:
            count += self.out_features
        return count
