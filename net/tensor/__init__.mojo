from tensor import TensorShape, TensorSpec
from tensor import Tensor as MojoTensor
from collections.optional import Optional, Variant
from net.checkpoint import fopen, tobytes, frombytes, Bytes, File
import math
from sys import exit
from .tutils import *
from .tensor import Tensor, tensor, ones, zeros, fill, rand
from net.kernel import scalar_op, tensor_op, Broadcast_op, vectorize, calculate_shapes, matmul, batch_matmul, randn, parallelize
from net.kernel import rand as rfill
from net.kernel.kmath import *
from memory import DTypePointer, memset
from builtin.io import _printf
from builtin.string import _calc_format_buffer_size