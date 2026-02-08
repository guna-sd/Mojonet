"""
AutoSGD: SGD optimizer for autograd Variables.

This optimizer works with the computation graph to update
parameters using their accumulated gradients.
"""

from .tensor import Tensor
from .graph import Graph
from .variable import Variable


struct AutoSGD:
    """Stochastic Gradient Descent optimizer for Variables.

    Updates parameters in a graph using their accumulated gradients.
    """

    var lr: Float32
    var param_ids: List[Int]  # IDs of parameters to optimize

    fn __init__(out self, lr: Float32 = 0.01):
        """Create SGD optimizer.

        Args:
            lr: Learning rate.
        """
        self.lr = lr
        self.param_ids = List[Int]()

    fn add_param(mut self, v: Variable):
        """Add a parameter to be optimized.

        Args:
            v: The Variable to optimize (must have requires_grad=True).
        """
        self.param_ids.append(v.id())

    fn step(self, mut graph: Graph):
        """Update all parameters using accumulated gradients.

        Performs: param = param - lr * grad
        """
        for i in range(len(self.param_ids)):
            var pid = self.param_ids[i]
            var data = graph.get_data(pid)
            var grad = graph.get_grad(pid)

            # Update: data = data - lr * grad
            for j in range(data.size()):
                data[j] = data[j] - self.lr * grad[j]

            graph.set_data(pid, data^)

    fn zero_grad(self, mut graph: Graph):
        """Zero gradients of all parameters."""
        for i in range(len(self.param_ids)):
            var pid = self.param_ids[i]
            graph.nodes[pid].zero_grad()

    fn set_lr(mut self, lr: Float32):
        """Set learning rate."""
        self.lr = lr
