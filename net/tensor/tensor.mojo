@value
struct TensorType[T: DType]:
    """A Underlying Storage that represents the Tensor."""

    var data: DTypePointer[T]
    """
    The data is a pointer to a block of memory that holds the elements of the tensor.
    """
    var shape: shape
    """
    The shape is representing the dimensions of the tensor.
    """
    var device: String
    """
    The name of the device the tensor is stored in (default cpu).
    """

    fn __init__(inout self: Self):
        self.data = stack_allocation[0, T]()
        self.shape = shape()
        self.device = "cpu"

    fn __init__(inout self: Self, shapes: shape, device: String = "cpu"):
        self.shape = shapes
        self.device = device
        self.data = DTypePointer[T]().alloc(self.shape.num_elements)
        memset_zero(self.data, self.shape.num_elements)

    fn __init__(
        inout self: Self,
        shapes: shape,
        data: DTypePointer[T],
        device: String = "cpu",
    ):
        self.shape = shapes
        self.device = device
        self.data = DTypePointer[T]().alloc(self.shape.num_elements)
        memcpy(self.data, data, self.shape.num_elements)

    fn __copyinit__(inout self: Self, other: Self):
        self.shape = other.shape
        self.data = DTypePointer[T]().alloc(self.shape.num_elements)
        self.device = other.device
        memcpy(self.data, other.data, self.shape.num_elements)

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape
        self.data = existing.data
        self.device = existing.device


@value
struct Tensor[type: DType = DType.float32](
    AnyType, CollectionElement, EqualityComparable, Stringable
):
    """
    A tensor is a multi-dimensional array of elements.
    """

    alias Type = TensorType[type]

    var tensor: Self.Type
    """
    The data is a pointer to a block of memory that holds the elements of the tensor.
    """
    var grad: Variant[Tensor[type], NoneType]
    """
    The grad variable holds the gradient of the tensor. It is an optional instance of the TensorType struct.
    This gradient is computed during the backward pass if requires_grad is True.
    """
    
    var requires_grad: Bool
    """
    The requires_grad variable indicates whether the tensor requires gradient computation.
    If set to True, the tensor will track operations for gradient computation during backpropagation.
    """
    var grad_fn: Variant[Function, NoneType]
    """
    The grad_fn variable holds a reference to a function responsible for computing the gradients of the tensor
    during the backward pass. This function is part of the autograd mechanism.
    """

    fn __init__(inout self):
        self.tensor = TensorType[type]()
        self.requires_grad = False
        self.grad = None
        self.grad_fn = None

    fn __init__(
        inout self,
        *shapes: Int,
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: shape,
        data: DTypePointer[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shapes, data, device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shapes, device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        *shapes: Int,
        data: DTypePointer[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), data, device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: shape,
        data: List[Scalar[type]],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        var tensor_data = DTypePointer[type]().alloc(shapes.num_elements)
        if shapes.num_elements == data.__len__():
            for i in range(shapes.num_elements):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shapes, device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: shape,
        *data: Scalar[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        var tensor_data = DTypePointer[type]().alloc(shapes.num_elements)
        if shapes.num_elements == data.__len__():
            for i in range(shapes.num_elements):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shapes, device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: List[Int],
        *data: Scalar[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        var tensor_shape = shape(shapes)
        var tensor_data = DTypePointer[type]().alloc(tensor_shape.num_elements)
        if tensor_shape.num_elements == data.__len__():
            for i in range(tensor_shape.num_elements):
                tensor_data[i] = data[i]
        else:
            for i in range(data.__len__()):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: shape,
        value: Scalar[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shapes, device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shapes, device)
        else:
            self.grad = None
        self = self.fill(value)

    fn __init__(
        inout self,
        shapes: VariadicList[Int],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self,
        shapes: List[Int],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self: Self,
        shapes: shape,
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shapes, device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shapes, device)
        else:
            self.grad = None

    fn __init__(
        inout self: Self,
        shapes: TensorShape,
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self: Self,
        shapes: TensorSpec,
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(shapes), device)
        else:
            self.grad = None

    fn __init__(
        inout self: Self,
        data: MojoTensor[type],
        device: String = "cpu",
        requires_grad : Bool = False,
    ):
        self.tensor = TensorType[type](shape(data._spec), data._ptr, device)
        self.requires_grad = requires_grad
        self.grad_fn = None
        if self.requires_grad:
            self.grad = TensorType[type](shape(data._spec), device)
        else:
            self.grad = None

    fn __copyinit__(inout self: Self, other: Self):
        self.tensor = other.tensor
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.grad_fn = other.grad_fn

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.tensor = existing.tensor
        self.requires_grad = existing.requires_grad
        self.grad = existing.grad
        self.grad_fn = existing.grad_fn

    @always_inline("nodebug")
    fn load[nelts: Int](self, owned index: Int) -> SIMD[type, nelts]:
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to load.

        Args:
            index : The index in the tensor data from which to load the SIMD value. If negative, it is interpreted as an index from the end of the data.

        Returns:
            The SIMD value loaded from the tensor data at the specified index.
        """
        if index < 0:
            index = self.num_elements() + index
        return self.tensor.data.load[width=nelts](index)

    @always_inline("nodebug")
    fn store[nelts: Int](self, owned index: Int, value: SIMD[type, nelts]):
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to store.

        Args:
            index : The index in the tensor data at which to store the SIMD value. If negative, it is interpreted as an index from the end of the data.
            value : The SIMD value to store in the tensor data at the specified index.
        """
        if index < 0:
            index = self.num_elements() + index
        self.tensor.data.store[width=nelts](index, value)

    @always_inline("nodebug")
    fn load(self, index: Int) -> SIMD[type, 1]:
        return self.load[1](index)

    @always_inline("nodebug")
    fn load[nelts: Int](self, *indices: Int) -> SIMD[type, nelts]:
        var pos = self.tensor.shape.offset(list(indices))
        return self.load[nelts](pos)

    @always_inline("nodebug")
    fn load(self, *indices: Int) -> SIMD[type, 1]:
        var pos = self.tensor.shape.offset(list(indices))
        return self.load[1](pos)

    @always_inline("nodebug")
    fn store(self: Self, index: Int, val: SIMD[type, 1]):
        self.store[1](index, val)

    @always_inline("nodebug")
    fn store[nelts: Int](self: Self, *indices: Int, val: SIMD[type, nelts]):
        var pos = self.tensor.shape.offset(list(indices))
        self.store[nelts](pos, val)

    @always_inline("nodebug")
    fn store(self: Self, *indices: Int, val: SIMD[type, 1]):
        var pos = self.tensor.shape.offset(list(indices))
        self.store(pos, val)

    @always_inline("nodebug")
    fn store(self: Self, indices: List[Int], val: SIMD[type, 1]):
        var pos = self.tensor.shape.offset(indices)
        self.store(pos, val)

    fn __getitem__(self: Self, index: Int) -> SIMD[type, 1]:
        return self.load[1](index)

    fn __getattr__[index: Int](self: Self) -> SIMD[type, 1]:
        return self.load[1](index)

    fn __getitem__(self, *indices: Int) -> SIMD[type, 1]:
        var pos = self.tensor.shape.offset(list(indices))
        return self.load[1](pos)

    fn __getitem__[*indices: Int](self) -> SIMD[type, 1]:
        var pos = self.tensor.shape.offset(list(indices))
        return self.load[1](pos)

    fn __getitem__(self, indices: List[Int]) -> SIMD[type, 1]:
        var pos = self.tensor.shape.offset(indices)
        return self.load[1](pos)

    fn __setitem__(self: Self, index: Int, val: SIMD[type, 1]):
        self.store(index, val)

    fn __setitem__(self: Self, *indices: Int, val: SIMD[type, 1]):
        var pos = self.tensor.shape.offset(list(indices))
        self[pos] = val

    fn __setitem__(self: Self, indices: List[Int], val: SIMD[type, 1]):
        var pos = self.tensor.shape.offset(indices)
        self[pos] = val

    @always_inline("nodebug")
    fn __eq__(self: Self, other: Self) -> Bool:
        var val = False
        if (
            self.num_elements() == other.num_elements()
            and self.tensor.shape == other.tensor.shape
        ):
            for i in range(self.num_elements()):
                if self.tensor.data[i] == other.tensor.data[i]:
                    val = True
        return val

    @always_inline("nodebug")
    fn __eq__(self: Self, other: MojoTensor[type]) -> Bool:
        var val = False
        if (
            self.num_elements() == other.num_elements()
            and self.tensor.shape == other.shape()
        ):
            for i in range(self.num_elements()):
                if self[i] == other[i]:
                    val = True
        return val

    @always_inline("nodebug")
    fn __ne__(self: Self, other: Self) -> Bool:
        if self.__eq__(other):
            return False
        return True

    @always_inline("nodebug")
    fn __ne__(self: Self, other: MojoTensor[type]) -> Bool:
        if self.__eq__(other):
            return False
        return True

    @always_inline("nodebug")
    fn __add__(self: Self, other: Self) -> Self:
        if self.tensor.shape == other.tensor.shape:
            return tensor_op[type, add](self, other)
        else:
            return Broadcast_op[type, add](self, other)

    @always_inline("nodebug")
    fn __add__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, add](self, other)

    @always_inline("nodebug")
    fn __radd__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, add](self, other)

    @always_inline("nodebug")
    fn __iadd__(inout self: Self, other: Self):
        if self.tensor.shape == other.tensor.shape:
            self = tensor_op[type, add](self, other)
        else:
            self = Broadcast_op[type, add](self, other)

    @always_inline("nodebug")
    fn __iadd__(inout self: Self, other: SIMD[type, 1]):
        self = scalar_op[type, add](self, other)

    @always_inline("nodebug")
    fn __sub__(self: Self, other: Self) -> Self:
        if self.tensor.shape == other.tensor.shape:
            return tensor_op[type, sub](self, other)
        else:
            return Broadcast_op[type, sub](self, other)

    @always_inline("nodebug")
    fn __sub__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, sub](self, other)

    @always_inline("nodebug")
    fn __rsub__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, sub](self, other)

    @always_inline("nodebug")
    fn __isub__(inout self: Self, other: Self):
        if self.tensor.shape == other.tensor.shape:
            self = tensor_op[type, sub](self, other)
        else:
            self = Broadcast_op[type, sub](self, other)

    @always_inline("nodebug")
    fn __isub__(inout self: Self, other: SIMD[type, 1]):
        self = scalar_op[type, sub](self, other)

    @always_inline("nodebug")
    fn __mul__(self: Self, other: Self) -> Self:
        if self.tensor.shape == other.tensor.shape:
            return tensor_op[type, mul](self, other)
        else:
            return Broadcast_op[type, mul](self, other)

    @always_inline("nodebug")
    fn __mul__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, mul](self, other)

    @always_inline("nodebug")
    fn __rmul__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, mul](self, other)

    @always_inline("nodebug")
    fn __imul__(inout self: Self, other: Self):
        if self.tensor.shape == other.tensor.shape:
            self = tensor_op[type, mul](self, other)
        else:
            self = Broadcast_op[type, mul](self, other)

    @always_inline("nodebug")
    fn __imul__(inout self: Self, other: SIMD[type, 1]):
        self = scalar_op[type, mul](self, other)

    @always_inline("nodebug")
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.tensor.shape == other.tensor.shape:
            return tensor_op[type, div](self, other)
        else:
            return Broadcast_op[type, div](self, other)

    @always_inline("nodebug")
    fn __truediv__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, div](self, other)

    @always_inline("nodebug")
    fn __rtruediv__(self: Self, other: SIMD[type, 1]) -> Self:
        return scalar_op[type, div](self, other)

    @always_inline("nodebug")
    fn __itruediv__(inout self: Self, other: Self):
        if self.tensor.shape == other.tensor.shape:
            self = tensor_op[type, div](self, other)
        else:
            self = Broadcast_op[type, div](self, other)

    @always_inline("nodebug")
    fn __itruediv__(inout self: Self, other: SIMD[type, 1]):
        self = scalar_op[type, div](self, other)

    @always_inline("nodebug")
    fn __neg__(self: Self) -> Self:
        return self.multiply(Scalar[type](-1))

    @always_inline("nodebug")
    fn __pow__(self: Self, exponent: Int) -> Self:
        """
        Exponentiation of each element in the tensor by the given exponent.
        """
        var result = self

        @parameter
        fn power[nelts: Int](i: Int):
            result.tensor.data[i] = pow(self.tensor.data[i], exponent)

        vectorize[power, 1](self.num_elements())
        return result

    @always_inline("nodebug")
    fn __ipow__(inout self: Self, exponent: Int):
        """
        In-place exponentiation of each element in the tensor by the given exponent.
        """
        self = self.__pow__(exponent)

    @always_inline
    fn __matmul__(self: Self, other: Self) -> Self:
        """
        Implements matrix multiplication for Tensor.
        The operation is defined as self @ other.
        """
        if self.rank() > 2 and other.rank() == 2:
            return matmul(self, other)
        if self.rank() == 2 and other.rank() == 2:
            return matmul(self, other)
        if self.rank() > 2 and other.rank() > 2:
            return batch_matmul[type](self, other)
        else:
            return batch_matmul(self, other)

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self^

    # TODO: write a fn for handling tensor Formatting with BUILTIN Formatter support and use here...
    fn __repr__(self: Self) -> String:
        var output = String()
        _ = output._unsafe_to_formatter()
        return output

    fn __str__[
        print_dtype: Bool = True, print_shape: Bool = True
    ](self: Self) -> String:
        return Tensorprinter[tprint, type, print_dtype, print_shape](
            self.tensor.data, self.shapes()
        )

    @always_inline("nodebug")
    fn apply[
        func: fn[dtype: DType, nelts: Int] (SIMD[dtype, nelts]) -> SIMD[
            dtype, nelts
        ]
    ](self: Self) -> Self:
        """
        A method for applying a function or an custom operation to the current Tensor.

        Parameters:
            func : `fn[dtype, nelts](SIMD[dtype, nelts])`.

        \tExample:
        ```mojo
        from net import Tensor
        from math import erf

        fn main():
            var x = Tensor[DType.bfloat16](2,2)
            x.rand()
            print(x.apply[erf]())
        ```
        """
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn operation[nelts: Int](idx: Int):
            self.store(idx, func(self.load(idx)))

        vectorize[operation, nelts, unroll_factor=4](self.num_elements())

        for i in range(self.num_elements() - (self.num_elements() % nelts)):
            if i >= self.num_elements():
                break
            if i % nelts == 0:
                continue
            self.store(i, func(self.load(i)))
        return self

    @always_inline("nodebug")
    fn apply[
        func: fn[dtype: DType, nelts: Int] (
            SIMD[dtype, nelts], SIMD[dtype, nelts]
        ) -> SIMD[dtype, nelts]
    ](self: Self, other: Self) -> Self:
        """
        A method for applying a function or an custom operation to the current Tensor.

        Parameters:
            func : `fn[dtype, nelts](SIMD[dtype, nelts], SIMD[dtype, nelts])`.

        Arguments:
            other : A Tensor with the same dtype and shape as the current Tensor.

        \tExample:
        ```mojo
        from net import Tensor
        from algorithm import vectorize

        fn add[type : DType, nelts : Int](first: SIMD[type,nelts], second : SIMD[type,nelts]) -> SIMD[type,nelts]:
            var result = SIMD[type,nelts]()
            @parameter
            fn addition[nelts : Int](i : Int):
                result[i] = first[i] + second[i]
            vectorize[addition,nelts](nelts)
            return result

        fn main():
            var x = Tensor[DType.bfloat16](2,2)
            x.rand()
            var y = Tensor[DType.bfloat16](2,2)
            y.rand()
            print(x.apply[add](y))
        ```
        """
        if self.tensor.shape != other.tensor.shape:
            print("shapes mismatch should not be different shapes")
            exit(1)
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn operation[nelts: Int](idx: Int):
            self.store(idx, func(self.load(idx), other.load(idx)))

        vectorize[operation, nelts, unroll_factor=4](self.num_elements())

        for i in range(self.num_elements() - (self.num_elements() % nelts)):
            if i >= self.num_elements():
                break
            if i % nelts == 0:
                continue
            self.store(i, func(self.load(i), other.load(i)))
        return self

    @always_inline("nodebug")
    fn pow(inout self: Self, pow: Int):
        self = self.__pow__(pow)

    @always_inline("nodebug")
    fn add(self: Self, x: SIMD[type, 1]) -> Self:
        return self.__add__(x)

    @always_inline("nodebug")
    fn add(self: Self, other: Tensor[type]) -> Self:
        return self.__add__(other)

    @always_inline("nodebug")
    fn sub(self: Self, x: SIMD[type, 1]) -> Self:
        return self.__sub__(x)

    @always_inline("nodebug")
    fn sub(self: Self, other: Tensor[type]) -> Self:
        return self.__sub__(other)

    @always_inline("nodebug")
    fn multiply(self: Self, x: SIMD[type, 1]) -> Self:
        return self.__mul__(x)

    @always_inline("nodebug")
    fn multiply(self: Self, other: Tensor[type]) -> Self:
        return self.__mul__(other)

    @always_inline("nodebug")
    fn div(self: Self, x: SIMD[type, 1]) -> Self:
        return self.__truediv__(x)

    fn div(self: Self, other: Tensor[type]) -> Self:
        return self.__truediv__(other)

    @always_inline("nodebug")
    fn sum(self: Self) -> Scalar[type]:
        """
        Compute the sum of all elements in the tensor.

        Returns:
            The sum of all elements in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn _sum[nelts: Int](i: Int):
            result += self[i].reduce_add()

        vectorize[_sum, nelts](self.num_elements())
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result += self[i]

        return result

    @always_inline("nodebug")
    fn max(self: Self) -> Scalar[type]:
        """
        Find the maximum value in the tensor.

        Returns:
            The maximum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn _max[nelts: Int](i: Int):
            result = max(result, self[i])

        vectorize[_max, nelts](self.num_elements())
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result = max(result, self[i])

        return result

    @always_inline("nodebug")
    fn min(self: Self) -> Scalar[type]:
        """
        Find the minimum value in the tensor.

        Returns:
            The minimum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn _min[nelts: Int](i: Int):
            result = min(result, self[i])

        vectorize[_min, nelts](self.num_elements())
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result = min(result, self[i])

        return result

    @always_inline("nodebug")
    fn mean(self: Self) -> Scalar[type]:
        """
        Compute the mean (average) value of the tensor.

        Returns:
            The mean value of the tensor as a scalar value.
        """
        return self.sum() / Scalar[type](self.num_elements())

    @always_inline("nodebug")
    fn prod(self: Self) -> Scalar[type]:
        """
        Compute the product of all elements in the tensor.

        Returns:
            The product of all elements in the tensor as a scalar value.
        """
        var result = Scalar[type](1)
        alias nelts = simdwidthof[type]() * 2

        @parameter
        fn _prod[nelts: Int](i: Int):
            result *= self[i].reduce_mul()

        vectorize[_prod, nelts](self.num_elements())
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result *= self[i]
        return result

    @always_inline("nodebug")
    fn relu(inout self : Self) -> Self:
        return relu[type](self)

    @always_inline("nodebug")
    fn tanh(inout self : Self) -> Self:
        return tanh[type](self)

    @always_inline("nodebug")
    fn gelu(inout self : Self) -> Self:
        return gelu[type](self)

    @always_inline("nodebug")
    fn silu(inout self : Self) -> Self:
        return silu[type](self)

    @always_inline("nodebug")
    fn list(self) -> List[Scalar[type]]:
        var result = List[Scalar[type]]()
        for i in range(self.num_elements()):
            result.append(self.load(i))
        return result

    @always_inline("nodebug")
    fn arange(self, start: Int, end: Int = 0, step: Int = 1) -> Tensor[type]:
        """
        Returns a tensor with values from start to end with specified step size.

        Args:
            start: The start value of the sequence.
            end: The end value of the sequence.
            step: The step size between consecutive values. Default is 1.

        Returns:
            A tensor containing the values from start to end with the specified step size.
        """
        var result = Tensor[type](self.shapes())
        var value = start

        @parameter
        fn arng[nelts: Int](i: Int):
            result[i] = value
            value += step

        if end == 0:
            vectorize[arng, 1](self.num_elements())
        vectorize[arng, 1](end)
        return result

    @always_inline("nodebug")
    fn arange(inout self):
        self = self.arange(0, self.num_elements())

    @always_inline("nodebug")
    fn zeros(self: Self):
        memset_zero(self.tensor.data, self.num_elements())

    fn zero_grad(inout self):
        """
        Zeroes the gradient of the tensor if it requires gradients.
        """
        if self.requires_grad:
            if not self.grad.isa[NoneType]():
                self.grad = self.grad.take[Tensor[type]]().fill(0)

    @always_inline
    fn sgd_update(inout self: Self, learning_rate: Float64):
        """
        Performs a gradient descent update on the tensor parameters.
        """
        if self.requires_grad:
            if not self.grad.isa[NoneType]():
                self -= self.grad.take[Tensor[type]]() * learning_rate

    fn backward(inout self, owned grad_output: Optional[Tensor[type]] = None) raises:
        """
        Perform the backward pass starting from this tensor, propagating gradients using the grad_fn if present.
        """
        if not self.requires_grad:
            print("Backward called on a tensor that does not require gradients.")
        
        if grad_output is None:
            if not self.grad.isa[NoneType]():
                self.grad = self.grad.take[Tensor[type]]().fill(0.0)
        
        if not self.grad_fn.isa[NoneType]():
            var func = self.grad_fn.take[Function]()
            func.invoke[type](self, grad_output.take())


    @always_inline("nodebug")
    fn ones(self: Self):
        memset[type](self.tensor.data, 1, self.num_elements())

    @always_inline("nodebug")
    fn rand(self):
        rfill[type](self.tensor.data, self.num_elements())
        # random.randn[type](self.data, self.shape.num_elements, 2, self.num_elements())

    @always_inline("nodebug")
    fn random(self) -> Self:
        rfill(self.tensor.data, self.num_elements())
        return self

    @always_inline("nodebug")
    fn fill(self: Self, value: Scalar[type]) -> Self:
        """Fill the tensor with a specified value."""
        var result = DTypePointer[type]().alloc(self.num_elements())

        @parameter
        fn _set(index: Int):
            result.store(index, self.load(index).splat(value))

        parallelize[_set](self.num_elements(), self.num_elements())
        return Self(self.tensor.shape, result)

    @always_inline("nodebug")
    fn ifill(inout self: Self, value: Scalar[type]):
        """Fill the tensor with a specified value."""
        self = self.fill(value)

    @always_inline("nodebug")
    fn transposed(self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(
                Error(
                    "dim1 and dim2 must be within the range of the tensor's"
                    " rank"
                )
            )
            abort(external_call["exit", Int](1))

        var tshape = self.shapes()
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        var ttensor = Tensor[type](tshape)

        for index in range(self.num_elements()):
            var _indices = self.shapes().indices(index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]
            ttensor[tindices] = self[index]

        return ttensor

    @always_inline("nodebug")
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var ttensor = self.transposed(dim1, dim2)
        self = Self(ttensor.shapes(), ttensor.tensor.data)

    @always_inline("nodebug")
    fn reshape(self: Self, other: shape) -> Self:
        """Reshape the tensor to the new dimensions and returns the reshaped tensor.
        """
        if self.tensor.shape.num_elements != other.num_elements:
            print("Error: Cannot reshape tensor.")
            exit(1)

        var data = Tensor[type](other)

        @parameter
        fn _reshape[nelts: Int](idx: Int):
            var old_indices = self.tensor.shape.indices(idx)
            var new_indices = other.indices(idx)
            data[new_indices] = self[old_indices]

        vectorize[_reshape, 1](self.num_elements())
        return Self(other, data.tensor.data)

    fn broadcast_to(inout self: Self, other: shape):
        var temp = Tensor[type](other)
        temp.zeros()
        self = Broadcast_op[type, add](self, temp)

    @always_inline("nodebug")
    fn reshape(self: Self, *new: Int) -> Self:
        """
        Reshape the tensor to the specified new shape.

        Args:
            new: The new shape dimensions as variadic integers.

        Returns:
            A new tensor reshaped to the new dimensions specified.
        """
        return self.reshape(shape(new))

    @always_inline("nodebug")
    fn reshape_to(inout self: Self, other: shape):
        """Reshape the tensor to the new dimensions."""
        self = Self(other, self.reshape(other).tensor.data)

    @always_inline("nodebug")
    fn flatten(self: Self) -> Self:
        """
        Flatten the tensor into a 1D array.

        Returns:
            A new tensor with all elements in a single dimension.
        """
        var new_shape = shape(self.tensor.shape.num_elements)
        return self.reshape(new_shape)

    @always_inline("nodebug")
    fn rank(self: Self) -> Int:
        return self.tensor.shape.ndim

    @always_inline("nodebug")
    fn shapes(self: Self) -> shape:
        return self.tensor.shape

    @always_inline("nodebug")
    fn data(self: Self) -> DTypePointer[type]:
        return self.tensor.data

    @always_inline("nodebug")
    fn dtype(self: Self) -> String:
        return type.__str__()

    #TODO: Add support for CUDA
    @always_inline("nodebug")
    fn device(self: Self) -> String:
        return self.tensor.device

    @always_inline("nodebug")
    fn num_elements(self: Self) -> Int:
        return self.tensor.shape.num_elements

    @always_inline("nodebug")
    fn astype[dtype: DType](self: Self) -> Tensor[dtype]:
        var casted = Tensor[dtype](self.tensor.shape)
        alias nelts = simdwidthof[dtype]() * 2

        @parameter
        fn cast_single_element[nelts: Int](index: Int):
            casted.store[nelts](index, self[index].cast[dtype]())

        vectorize[cast_single_element, nelts](self.num_elements())
        for index in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            casted[index] = self[index].cast[dtype]()
        return casted

    @always_inline("nodebug")
    fn num_bytes(self: Self) -> Int:
        return sizeof[type]() * self.num_elements()

    @always_inline("nodebug")
    fn itemsize(self: Self) -> Int:
        return sizeof[type]()

    # TODO: Add support for seralizing and deserializing tensors
    @always_inline("nodebug")
    fn serialize(self) -> List[Bytes]:
        var bytes = List[Bytes](capacity=self.num_elements())
        for i in range(self.num_elements()):
            bytes.append(tobytes[type](self.tensor.data[i]))
        return bytes

    @always_inline("nodebug")
    fn deserialize(self, data: List[Bytes], shape: List[Int]) -> Tensor[type]:
        var num_elements = num_elements(shape)
        var tensor_data = List[Scalar[type]](capacity=num_elements)
        for i in range(num_elements):
            var value = frombytes[type](data[i])
            tensor_data.append(value)
        return Tensor[type](shape, tensor_data)


fn tensor[
    dtype: DType = DType.float32
](Shape: List[Int], rand: Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor


fn tensor[
    dtype: DType = DType.float32
](*Shape: Int, rand: Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor


fn ones[
    dtype: DType = DType.float32
](Shape: List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](1))
    return tensor


fn ones[
    dtype: DType = DType.float32
](*Shape: Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](1))
    return tensor


fn zeros[
    dtype: DType = DType.float32
](Shape: List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](0))
    return tensor


fn zeros[
    dtype: DType = DType.float32
](*Shape: Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](0))
    return tensor


@always_inline("nodebug")
fn fill[
    dtype: DType = DType.float32
](shape: shape, val: Scalar[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](shape)
    alias nelts = simdwidthof[dtype]()

    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        result.store[nelts](idx, result.load[nelts](idx).splat(val))

    vectorize[fill_vec, nelts](result.num_elements())
    return result


@always_inline("nodebug")
fn fill[
    dtype: DType = DType.float32
](inout x: Tensor[dtype], val: Scalar[dtype]):
    x.ifill(val)


fn fill[
    dtype: DType = DType.float32
](*Shape: Int, val: Scalar[dtype]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, val)
    return tensor


fn fill[
    dtype: DType = DType.float32
](Shape: List[Int], val: Scalar[dtype]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, val)
    return tensor


fn rand[
    dtype: DType = DType.float32
](Shape: List[Int], seed: Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand()
    tensor.rand()
    return tensor


fn rand[
    dtype: DType = DType.float32
](*Shape: Int, seed: Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand()
    tensor.rand()
    return tensor


fn empty[dtype: DType = DType.float32](*Shape: Int) -> Tensor[dtype]:
    return Tensor[dtype](Shape)


fn empty[dtype: DType = DType.float32](Shape: List[Int]) -> Tensor[dtype]:
    return Tensor[dtype](Shape)
