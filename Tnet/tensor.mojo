"""
Tensor: Float32 tensor with basic operations.

Minimal implementation for the demo, compatible with Mojo 0.26.2.
"""

from memory import UnsafePointer, memset_zero, memcpy, alloc
from random import random_float64


struct Tensor(Copyable, Movable, Stringable, Writable):
    """Float32 tensor with shape and basic operations."""

    var _data: UnsafePointer[Float32, MutAnyOrigin]
    var _shape: List[Int]
    var _size: Int
    var requires_grad: Bool

    # ═══════════════════════════════════════════════════════════════════════
    # Construction
    # ═══════════════════════════════════════════════════════════════════════

    fn __init__(out self):
        """Create an empty tensor."""
        self._data = UnsafePointer[Float32, MutAnyOrigin]()
        self._shape = List[Int]()
        self._size = 0
        self.requires_grad = False

    fn __init__(out self, *shapes: Int, requires_grad: Bool = False):
        """Create a zero-initialized tensor with given shape."""
        var shape_list = List[Int]()
        var size = 1
        for i in range(len(shapes)):
            shape_list.append(shapes[i])
            size *= shapes[i]

        self._shape = shape_list^
        self._size = size
        self._data = alloc[Float32](size)
        memset_zero(self._data, size)
        self.requires_grad = requires_grad

    fn __init__(out self, var shape: List[Int], requires_grad: Bool = False):
        """Create a zero-initialized tensor from shape list."""
        var size = 1
        for i in range(len(shape)):
            size *= shape[i]

        self._shape = shape^
        self._size = size
        self._data = alloc[Float32](size)
        memset_zero(self._data, size)
        self.requires_grad = requires_grad

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self._data = existing._data
        self._shape = existing._shape^
        self._size = existing._size
        self.requires_grad = existing.requires_grad

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self._shape = existing._shape.copy()
        self._size = existing._size
        self.requires_grad = existing.requires_grad
        self._data = alloc[Float32](self._size)
        memcpy(dest=self._data, src=existing._data, count=self._size)

    fn __del__(deinit self):
        """Destructor - free memory."""
        if self._data:
            self._data.free()

    fn copy(self) -> Self:
        """Explicit copy."""
        var result = Self()
        result._shape = self._shape.copy()
        result._size = self._size
        result.requires_grad = self.requires_grad
        result._data = alloc[Float32](self._size)
        memcpy(dest=result._data, src=self._data, count=self._size)
        return result^

    # ═══════════════════════════════════════════════════════════════════════
    # Indexing
    # ═══════════════════════════════════════════════════════════════════════

    fn __getitem__(self, idx: Int) -> Float32:
        """Get element at flat index."""
        return self._data[idx]

    fn __setitem__(mut self, idx: Int, value: Float32):
        """Set element at flat index."""
        self._data[idx] = value

    # ═══════════════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════════════

    fn size(self) -> Int:
        """Total number of elements."""
        return self._size

    fn shape(self) -> List[Int]:
        """Shape of tensor."""
        return self._shape.copy()

    fn numel(self) -> Int:
        """Alias for size()."""
        return self._size

    # ═══════════════════════════════════════════════════════════════════════
    # Arithmetic Operations
    # ═══════════════════════════════════════════════════════════════════════

    fn __add__(self, other: Self) raises -> Self:
        """Element-wise addition."""
        if self._size != other._size:
            raise Error("Tensor size mismatch in add")
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] + other._data[i]
        return result^

    fn __sub__(self, other: Self) raises -> Self:
        """Element-wise subtraction."""
        if self._size != other._size:
            raise Error("Tensor size mismatch in sub")
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] - other._data[i]
        return result^

    fn __mul__(self, other: Self) raises -> Self:
        """Element-wise multiplication."""
        if self._size != other._size:
            raise Error("Tensor size mismatch in mul")
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] * other._data[i]
        return result^

    fn __mul__(self, scalar: Float32) -> Self:
        """Scalar multiplication."""
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] * scalar
        return result^

    fn __truediv__(self, scalar: Float32) -> Self:
        """Scalar division."""
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] / scalar
        return result^

    fn __pow__(self, exp: Float32) -> Self:
        """Element-wise power."""
        var result = Tensor(self._shape.copy())
        for i in range(self._size):
            result._data[i] = self._data[i] ** exp
        return result^

    # ═══════════════════════════════════════════════════════════════════════
    # Matrix Operations
    # ═══════════════════════════════════════════════════════════════════════

    fn matmul(self, other: Self) raises -> Self:
        """Matrix multiplication: self @ other."""
        if len(self._shape) != 2 or len(other._shape) != 2:
            raise Error("matmul requires 2D tensors")
        if self._shape[1] != other._shape[0]:
            raise Error("matmul shape mismatch")

        var m = self._shape[0]
        var k = self._shape[1]
        var n = other._shape[1]

        var result_shape = List[Int]()
        result_shape.append(m)
        result_shape.append(n)
        var result = Tensor(result_shape^)

        for i in range(m):
            for j in range(n):
                var sum: Float32 = 0.0
                for p in range(k):
                    sum += self._data[i * k + p] * other._data[p * n + j]
                result._data[i * n + j] = sum

        return result^

    fn T(self) raises -> Self:
        """Transpose (2D only)."""
        if len(self._shape) != 2:
            raise Error("transpose requires 2D tensor")

        var rows = self._shape[0]
        var cols = self._shape[1]

        var new_shape = List[Int]()
        new_shape.append(cols)
        new_shape.append(rows)
        var result = Tensor(new_shape^)

        for i in range(rows):
            for j in range(cols):
                result._data[j * rows + i] = self._data[i * cols + j]

        return result^

    # ═══════════════════════════════════════════════════════════════════════
    # Reductions
    # ═══════════════════════════════════════════════════════════════════════

    fn sum(self) -> Float32:
        """Sum all elements."""
        var result: Float32 = 0.0
        for i in range(self._size):
            result += self._data[i]
        return result

    fn mean(self) -> Float32:
        """Mean of all elements."""
        return self.sum() / Float32(self._size)

    fn item(self) -> Float32:
        """Get single element (for scalar tensors)."""
        return self._data[0]

    # ═══════════════════════════════════════════════════════════════════════
    # In-place operations
    # ═══════════════════════════════════════════════════════════════════════

    fn zero_(mut self):
        """Zero all elements in-place."""
        memset_zero(self._data, self._size)

    fn fill_(mut self, value: Float32):
        """Fill with value in-place."""
        for i in range(self._size):
            self._data[i] = value

    fn add_(mut self, other: Self):
        """In-place addition."""
        for i in range(self._size):
            self._data[i] += other._data[i]

    fn sub_(mut self, other: Self):
        """In-place subtraction."""
        for i in range(self._size):
            self._data[i] -= other._data[i]

    # ═══════════════════════════════════════════════════════════════════════
    # String representation
    # ═══════════════════════════════════════════════════════════════════════

    fn __str__(self) -> String:
        """String representation."""
        var result = String("Tensor([")
        var max_show = 6
        var show = min(self._size, max_show)
        for i in range(show):
            if i > 0:
                result += ", "
            result += String(self._data[i])
        if self._size > max_show:
            result += ", ..."
        result += "], shape=["
        for i in range(len(self._shape)):
            if i > 0:
                result += ", "
            result += String(self._shape[i])
        result += "])"
        return result

    fn write_to[W: Writer](self, mut writer: W):
        """Write to writer."""
        writer.write(self.__str__())


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════


fn zeros1d(d0: Int) -> Tensor:
    """Create 1D zero-filled tensor."""
    return Tensor(d0)


fn zeros(d0: Int, d1: Int) -> Tensor:
    """Create 2D zero-filled tensor."""
    return Tensor(d0, d1)


fn ones1d(d0: Int) -> Tensor:
    """Create 1D ones-filled tensor."""
    var t = Tensor(d0)
    t.fill_(1.0)
    return t^


fn ones(d0: Int, d1: Int) -> Tensor:
    """Create 2D ones-filled tensor."""
    var t = Tensor(d0, d1)
    t.fill_(1.0)
    return t^


fn random1d(d0: Int) -> Tensor:
    """Create 1D random tensor [0, 1)."""
    var t = Tensor(d0)
    for i in range(t._size):
        t._data[i] = Float32(random_float64())
    return t^


fn random(d0: Int, d1: Int) -> Tensor:
    """Create 2D random tensor [0, 1)."""
    var t = Tensor(d0, d1)
    for i in range(t._size):
        t._data[i] = Float32(random_float64())
    return t^
