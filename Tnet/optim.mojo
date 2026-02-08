"""
SGD: Stochastic Gradient Descent optimizer.
"""

from .tensor import Tensor
from .linear import Linear


struct SGD:
    """Stochastic Gradient Descent optimizer."""

    var lr: Float32

    fn __init__(out self, lr: Float32 = 0.01):
        """Create SGD optimizer."""
        self.lr = lr

    fn step(self, mut layer: Linear):
        """Update parameters of a Linear layer."""
        # Update weights
        for i in range(layer.weight.size()):
            layer.weight[i] = layer.weight[i] - self.lr * layer.weight_grad[i]

        # Update bias
        if layer.has_bias:
            for i in range(layer.bias.size()):
                layer.bias[i] = layer.bias[i] - self.lr * layer.bias_grad[i]

    fn set_lr(mut self, lr: Float32):
        """Set learning rate."""
        self.lr = lr
