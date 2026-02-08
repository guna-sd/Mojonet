"""
Linear: Fully connected layer.

Implements y = x @ W.T + b with forward and backward pass.
"""

from .tensor import Tensor, random


struct Linear(Movable):
    """Fully connected layer: y = x @ W.T + b."""

    var weight: Tensor  # [out_features, in_features]
    var bias: Tensor  # [out_features]
    var in_features: Int
    var out_features: Int
    var has_bias: Bool

    # Saved for backward
    var _last_input: Tensor

    # Gradients
    var weight_grad: Tensor
    var bias_grad: Tensor

    fn __init__(
        out self, in_features: Int, out_features: Int, has_bias: Bool = True
    ):
        """Create linear layer."""
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # Initialize weight with small random values
        self.weight = random(out_features, in_features)
        # Scale to Xavier-like initialization
        var scale = Float32(1.0) / Float32(in_features)
        for i in range(self.weight.size()):
            self.weight[i] = (self.weight[i] - 0.5) * scale

        # Initialize bias to zero
        self.bias = Tensor(out_features)

        # Initialize gradients
        self.weight_grad = Tensor(out_features, in_features)
        self.bias_grad = Tensor(out_features)

        # Placeholder for saved input
        self._last_input = Tensor()

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.weight = existing.weight^
        self.bias = existing.bias^
        self.in_features = existing.in_features
        self.out_features = existing.out_features
        self.has_bias = existing.has_bias
        self._last_input = existing._last_input^
        self.weight_grad = existing.weight_grad^
        self.bias_grad = existing.bias_grad^

    # ═══════════════════════════════════════════════════════════════════════
    # Forward Pass
    # ═══════════════════════════════════════════════════════════════════════

    fn forward(mut self, x: Tensor) raises -> Tensor:
        """Forward pass: y = x @ W.T + b."""
        # Save input for backward (explicit copy)
        self._last_input = x.copy()

        # y = x @ W.T
        var wt = self.weight.T()
        var output = x.matmul(wt)

        # Add bias: y += b (broadcast over batch)
        if self.has_bias:
            var batch = output.shape()[0]
            for i in range(batch):
                for j in range(self.out_features):
                    output[i * self.out_features + j] += self.bias[j]

        return output^

    # ═══════════════════════════════════════════════════════════════════════
    # Backward Pass
    # ═══════════════════════════════════════════════════════════════════════

    fn backward(mut self, grad_output: Tensor) raises -> Tensor:
        """Backward pass: compute gradients."""
        var batch = self._last_input.shape()[0]

        # Gradient w.r.t. weight: dL/dW = grad_output.T @ input
        var grad_output_t = grad_output.T()
        var dw = grad_output_t.matmul(self._last_input)
        self.weight_grad.add_(dw)

        # Gradient w.r.t. bias: dL/db = sum(grad_output, axis=0)
        if self.has_bias:
            for j in range(self.out_features):
                var sum: Float32 = 0.0
                for i in range(batch):
                    sum += grad_output[i * self.out_features + j]
                self.bias_grad[j] = self.bias_grad[j] + sum

        # Gradient w.r.t. input: dL/dx = grad_output @ W
        var grad_input = grad_output.matmul(self.weight)

        return grad_input^

    # ═══════════════════════════════════════════════════════════════════════
    # Gradient Management
    # ═══════════════════════════════════════════════════════════════════════

    fn zero_grad(mut self):
        """Zero all gradients."""
        self.weight_grad.zero_()
        self.bias_grad.zero_()

    fn num_parameters(self) -> Int:
        """Total number of parameters."""
        var count = self.weight.size()
        if self.has_bias:
            count += self.bias.size()
        return count
