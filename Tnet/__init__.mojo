"""
Tnet: Minimal Deep Learning Module for MojoNet Demo.

A self-contained module providing:
- Tensor operations
- Manual Linear layer with forward/backward
- Graph-based automatic differentiation
- AutoLinear layer using autograd
- SGD optimizer (manual and autograd versions)
- Checkpointing

Compatible with Mojo 0.26.2.
"""

# Core tensor operations
from .tensor import Tensor, zeros, zeros1d, ones, ones1d, random, random1d

# Manual implementation (original demo)
from .linear import Linear
from .optim import SGD
from .checkpoint import save, load_into

# Graph-based autograd system
from .graph import Graph, GradContext, VariableNode
from .graph import OP_ADD, OP_SUB, OP_MUL, OP_MATMUL, OP_MEAN, OP_POW, OP_SUM

# Variable and differentiable operations
from .variable import Variable
from .variable import add, sub, mul, matmul, mean, sum_all, pow, scalar_mul
from .variable import broadcast_add, transpose

# Autograd-based modules
from .autolinear import AutoLinear
from .autooptim import AutoSGD
