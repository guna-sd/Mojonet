from net.utils.rand import rand_n
from net.utils.mt19937 import mt19937Engine
from memory import DTypePointer, memset, memset_zero, memcpy
from .utils import list


@value
@register_passable("trivial")
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
    var device: StringLiteral
    """
    The name of the device the tensor is stored in (default cpu).
    """

    fn __init__(inout self: Self):
        self.data = stack_allocation[0, T]()
        self.shape = shape()
        self.device = "cpu"

    fn __init__(inout self: Self, shapes: shape, device: StringLiteral = "cpu"):
        self.shape = shapes
        self.device = device
        self.data = DTypePointer[T]().alloc(self.shape.numofelements())
        memset_zero(self.data, self.shape.numofelements())

    fn __init__(
        inout self: Self,
        shapes: shape,
        data: DTypePointer[T],
        device: StringLiteral = "cpu",
    ):
        self.shape = shapes
        self.device = device
        self.data = DTypePointer[T]().alloc(self.shape.numofelements())
        memcpy(self.data, data, self.shape.numofelements())

    @always_inline("nodebug")
    fn load[nelts: Int](self, owned index: Int) -> SIMD[T, nelts]:
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to load.

        Args:
            index : The index in the tensor data from which to load the SIMD value. If negative, it is interpreted as an index from the end of the data.

        Returns:
            The SIMD value loaded from the tensor data at the specified index.
        """
        if index < 0:
            index = self.shape.num_elements + index
        return self.data.load[width=nelts](index)

    @always_inline("nodebug")
    fn store[nelts: Int](self, owned index: Int, value: SIMD[T, nelts]):
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to store.

        Args:
            index : The index in the tensor data at which to store the SIMD value. If negative, it is interpreted as an index from the end of the data.
            value : The SIMD value to store in the tensor data at the specified index.
        """
        if index < 0:
            index = self.shape.num_elements + index
        self.data.store[width=nelts](index, value)

    @always_inline("nodebug")
    fn load[nelts: Int = 1](self, *indices: Int) -> SIMD[T, nelts]:
        var pos = self.shape.offset(list(indices))
        return self.load[nelts](pos)

    @always_inline("nodebug")
    fn load[nelts: Int = 1](self, indices: List[Int]) -> SIMD[T, nelts]:
        var pos = self.shape.offset(indices)
        return self.load[nelts](pos)

    @always_inline("nodebug")
    fn store[nelts: Int = 1](self, indices: List[Int], val: SIMD[T, nelts]):
        var pos = self.shape.offset(indices)
        self.store[nelts](pos, val)

    @always_inline("nodebug")
    fn store[nelts: Int = 1](self, *indices: Int, val: SIMD[T, nelts]):
        var pos = self.shape.offset(list(indices))
        self.store[nelts](pos, val)

    @always_inline("nodebug")
    fn __getitem__(self: Self, index: Int) -> SIMD[T, 1]:
        return self.load[1](index)

    @always_inline("nodebug")
    fn __getitem__(self, *indices: Int) -> SIMD[T, 1]:
        var pos = self.shape.offset(list(indices))
        return self.load[1](pos)

    @always_inline("nodebug")
    fn __getitem__(self, indices: List[Int]) -> SIMD[T, 1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)

    @always_inline("nodebug")
    fn __setitem__(self: Self, index: Int, val: SIMD[T, 1]):
        self.store(index, val)

    @always_inline("nodebug")
    fn __setitem__(self: Self, *indices: Int, val: SIMD[T, 1]):
        var pos = self.shape.offset(list(indices))
        self[pos] = val

    @always_inline("nodebug")
    fn __setitem__(self: Self, indices: List[Int], val: SIMD[T, 1]):
        var pos = self.shape.offset(indices)
        self[pos] = val


@value
struct _TensorIter[
    T: DType,
    forward: Bool = True,
]:
    """Iterator for Tensor.

    Parameters:
        T: The type of the elements in the tensor.
        forward: The iteration direction. `False` is backwards.
    """

    alias type = Tensor[T]

    var index: Int
    var src: Self.type

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        inout self,
    ) -> Tensor[T]:
        @parameter
        if forward:
            self.index += 1
            return self.src[self.index - 1]
        else:
            self.index -= 1
            return self.src[self.index]

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src) - self.index
        else:
            return self.index


@value
struct Tensor[type: DType = DType.float32](
    Absable,
    Ceilable,
    CeilDivable,
    CollectionElement,
    EqualityComparable,
    Formattable,
    Floorable,
    Powable,
    Representable,
    Sized,
    Stringable,
    Truncable,
):
    """
    A tensor is a multi-dimensional array of elements.
    """

    alias Type = TensorType[type]
    alias Operation = Operation[type]

    var tensor: Self.Type
    """
    The data is a pointer to a block of memory that holds the elements of the tensor.
    """

    var requires_grad: Bool
    """
    The requires_grad variable indicates whether the tensor requires gradient computation.
    If set to True, the tensor will track operations for gradient computation during backpropagation.
    """

    fn __init__(inout self):
        self.tensor = TensorType[type]()
        self.requires_grad = False

    fn __init__(
        inout self,
        owned tensortype: TensorType[type],
        owned requires_grad: Bool = False,
    ):
        self.tensor = tensortype
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        *shapes: Int,
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        *shapes: Int,
        data: DTypePointer[type],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), data, device)
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        shapes: shape,
        data: List[Scalar[type]],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        var tensor_data = DTypePointer[type]().alloc(shapes.num_elements)
        if shapes.numofelements() == data.__len__():
            for i in range(shapes.numofelements()):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        shapes: shape,
        *data: Scalar[type],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        var tensor_data = DTypePointer[type]().alloc(shapes.num_elements)
        if shapes.numofelements() == data.__len__():
            for i in range(shapes.numofelements()):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        shapes: List[Int],
        *data: Scalar[type],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        var tensor_shape = shape(shapes)
        var tensor_data = DTypePointer[type]().alloc(tensor_shape.num_elements)
        if tensor_shape.numofelements() == data.__len__():
            for i in range(tensor_shape.numofelements()):
                tensor_data[i] = data[i]
        else:
            for i in range(data.__len__()):
                tensor_data[i] = data[i]
        self.tensor = TensorType[type](shapes, tensor_data, device)
        tensor_data.free()
        self.requires_grad = requires_grad

    fn __init__(
        inout self,
        shapes: shape,
        value: Scalar[type],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shapes, device)
        self.requires_grad = requires_grad
        self = self.fill(value)

    fn __init__(
        inout self,
        shapes: List[Int],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shape(shapes), device)
        self.requires_grad = requires_grad

    fn __init__(
        inout self: Self,
        shapes: shape,
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shapes, device)
        self.requires_grad = requires_grad

    fn __init__(
        inout self: Self,
        data: MojoTensor[type],
        device: StringLiteral = "cpu",
        requires_grad: Bool = False,
    ):
        self.tensor = TensorType[type](shape(data._spec), data._ptr, device)
        self.requires_grad = requires_grad

    fn __copyinit__(inout self: Self, other: Self):
        self.tensor = other.tensor
        self.requires_grad = other.requires_grad

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.tensor = existing.tensor
        self.requires_grad = existing.requires_grad

    fn __iter__(
        self,
    ) -> _TensorIter[type, _]:
        """Iterate over elements of the tensor."""
        return _TensorIter(0, self)

    fn __reversed__(
        self,
    ) -> _TensorIter[type, False]:
        """Iterate backwards over the tensor."""
        return _TensorIter[type, False](len(self), self)

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self^

    fn __len__(self) -> Int:
        return self.tensor.shape[0]

    fn __repr__(self: Self) -> String:
        var output = String()
        var writer = output._unsafe_to_formatter()
        self.format_to[True, True](writer)
        return output^

    fn __str__(self: Self) -> String:
        return String.format_sequence(self)

    @always_inline("nodebug")
    fn format_to(self, inout writer: Formatter):
        TensorPrinter[type](self.tensor.data, self.shapes(), writer)

    @always_inline("nodebug")
    fn format_to[
        print_type: Bool, print_shape: Bool
    ](self, inout writer: Formatter):
        TensorPrinter[type, print_type, print_shape](
            self.tensor.data, self.shapes(), writer
        )

    @always_inline("nodebug")
    fn load[nelts: Int = 1](self, index: Int) -> SIMD[type, nelts]:
        return self.tensor.load[nelts](index)

    @always_inline("nodebug")
    fn load[nelts: Int = 1](self, *indices: Int) -> SIMD[type, nelts]:
        return self.tensor.load[nelts](list(indices))

    @always_inline("nodebug")
    fn load[nelts: Int = 1](self, indices: List[Int]) -> SIMD[type, nelts]:
        return self.tensor.load[nelts](indices)

    @always_inline("nodebug")
    fn store[nelts: Int = 1](self: Self, *indices: Int, val: SIMD[type, nelts]):
        self.tensor.store[nelts](list(indices), val)

    @always_inline("nodebug")
    fn store[
        nelts: Int = 1
    ](self: Self, indices: List[Int], val: SIMD[type, nelts]):
        self.tensor.store[nelts](indices, val)

    @always_inline("nodebug")
    fn store[nelts: Int = 1](self: Self, index: Int, val: SIMD[type, nelts]):
        self.tensor.store[nelts](index, val)

    # TODO: based on the experience from pytorch the __getitem__ should probably return Tensor instead of a SIMD
    # working on it
    # @always_inline("nodebug")
    # fn __getitem__(self: Self, index: Int) -> Tensor[type]:
    #     var offset = self.tensor.shape.indices(index)
    #     var new_shape = self.tensor.shape.Shapes()[1:]
    #     var new_tensor_data = self.data(shape(new_shape), offset[0])
    #     return Tensor[type](TensorType[type](shape(new_shape), new_tensor_data, self.tensor.device))

    # @always_inline("nodebug")
    # fn __getitem__(self, *indices: Int) -> Tensor[type]:
    #     var index_list = list(indices)
    #     var tensor_data = self
    #     for index in index_list:
    #         tensor_data = tensor_data[index[]]
    #     return tensor_data

    @always_inline("nodebug")
    fn __getitem__(self: Self, index: Int) -> SIMD[type, 1]:
        return self.tensor[index]

    @always_inline("nodebug")
    fn __getitem__(self, *indices: Int) -> SIMD[type, 1]:
        return self.tensor[list(indices)]

    @always_inline("nodebug")
    fn __getitem__(self, indices: List[Int]) -> SIMD[type, 1]:
        return self.tensor[indices]

    @always_inline("nodebug")
    fn __setitem__(self: Self, index: Int, val: SIMD[type, 1]):
        self.tensor[index] = val

    @always_inline("nodebug")
    fn __setitem__(self: Self, *indices: Int, val: SIMD[type, 1]):
        self.tensor[list(indices)] = val

    @always_inline("nodebug")
    fn __setitem__(self: Self, indices: List[Int], val: SIMD[type, 1]):
        self.tensor[indices] = val

    @always_inline("nodebug")
    fn __eq__(self: Self, other: Self) -> Bool:
        if (
            self.num_elements() != other.num_elements()
            or self.tensor.shape != other.tensor.shape
        ):
            return False
        for i in range(self.num_elements()):
            if self.tensor.data[i] != other.tensor.data[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __eq__(self: Self, other: MojoTensor[type]) -> Bool:
        if (
            self.num_elements() != other.num_elements()
            or self.tensor.shape != other.shape()
        ):
            return False
        for i in range(self.num_elements()):
            if self.load(i) != other.load(i):
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self: Self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __ne__(self: Self, other: MojoTensor[type]) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __bool__(self: Self) -> Bool:
        if self.num_elements() > 0:
            return True
        else:
            return False

    @always_inline("nodebug")
    fn __add__(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__add__](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __sub__(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__sub__](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __mul__(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__mul__](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __truediv__(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__truediv__](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __floordiv__(self: Self, other: Self) -> Self:
        alias func: fn[Type: DType, nelts: Int] (
            SIMD[Type, nelts], SIMD[Type, nelts]
        ) -> SIMD[Type, nelts] = SIMD.__floordiv__
        var result = Self.Operation.tensor_operation[func](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __mod__(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__mod__](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __pow__(self: Self, exponent: Self) -> Self:
        """
        Exponentiation of each element in the tensor by the given exponent.
        """
        alias func: fn[Type: DType, nelts: Int] (
            SIMD[Type, nelts], SIMD[Type, nelts]
        ) -> SIMD[Type, nelts] = SIMD.__pow__
        var result = Self.Operation.tensor_operation[func](
            self.tensor, exponent.tensor
        )
        var requires_grad = self.requires_grad or exponent.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __add__(self: Self, other: SIMD[type, 1]) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__add__](
            self.tensor, other
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __sub__(self: Self, other: SIMD[type, 1]) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__sub__](
            self.tensor, other
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __mul__(self: Self, other: SIMD[type, 1]) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__mul__](
            self.tensor, other
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __truediv__(self: Self, other: SIMD[type, 1]) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__truediv__](
            self.tensor, other
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __floordiv__(self: Self, other: SIMD[type, 1]) -> Self:
        alias func: fn[Type: DType, nelts: Int] (
            SIMD[Type, nelts], SIMD[Type, nelts]
        ) -> SIMD[Type, nelts] = SIMD.__floordiv__
        var result = Self.Operation.tensor_operation[func](self.tensor, other)
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __mod__(self: Self, other: SIMD[type, 1]) -> Self:
        var result = Self.Operation.tensor_operation[SIMD.__mod__](
            self.tensor, other
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __pow__(self: Self, exponent: Int) -> Self:
        """
        Exponentiation of each element in the tensor by the given exponent.
        """
        alias func: fn[Type: DType, nelts: Int] (
            SIMD[Type, nelts], Int
        ) -> SIMD[Type, nelts] = SIMD.__pow__
        var result = Self.Operation.tensor_operation[func](
            self.tensor, exponent
        )
        var requires_grad = self.requires_grad or exponent

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __iadd__(inout self: Self, other: Self):
        self = self + other

    @always_inline("nodebug")
    fn __isub__(inout self: Self, other: Self):
        self = self - other

    @always_inline("nodebug")
    fn __imul__(inout self: Self, other: Self):
        self = self * other

    @always_inline("nodebug")
    fn __itruediv__(inout self: Self, other: Self):
        self = self / other

    @always_inline("nodebug")
    fn __ifloordiv__(inout self: Self, other: Self):
        self = self // other

    @always_inline("nodebug")
    fn __imod__(inout self: Self, other: Self):
        self = self % other

    @always_inline("nodebug")
    fn __ipow__(inout self: Self, exponent: Self):
        """
        In-place exponentiation of each element in the tensor by the given exponent.
        """
        constrained[type.is_numeric(), "the Tensor type must be numeric"]()
        self = self.__pow__(exponent)

    @always_inline("nodebug")
    fn __iadd__(inout self: Self, other: SIMD[type, 1]):
        self = self + other

    @always_inline("nodebug")
    fn __isub__(inout self: Self, other: SIMD[type, 1]):
        self = self - other

    @always_inline("nodebug")
    fn __imul__(inout self: Self, other: SIMD[type, 1]):
        self = self * other

    @always_inline("nodebug")
    fn __itruediv__(inout self: Self, other: SIMD[type, 1]):
        self = self / other

    @always_inline("nodebug")
    fn __ifloordiv__(inout self: Self, other: SIMD[type, 1]):
        self = self // other

    @always_inline("nodebug")
    fn __imod__(inout self: Self, other: SIMD[type, 1]):
        self = self % other

    @always_inline("nodebug")
    fn __ipow__(inout self: Self, exponent: Int):
        """
        In-place exponentiation of each element in the tensor by the given exponent.
        """
        self = self.__pow__(exponent)

    @always_inline("nodebug")
    fn __radd__(self: Self, other: Self) -> Self:
        return other + self

    @always_inline("nodebug")
    fn __rsub__(self: Self, other: Self) -> Self:
        return other - self

    @always_inline("nodebug")
    fn __rmul__(self: Self, other: Self) -> Self:
        return other * self

    @always_inline("nodebug")
    fn __rtruediv__(self: Self, other: Self) -> Self:
        return other / self

    @always_inline("nodebug")
    fn __rfloordiv__(self: Self, other: Self) -> Self:
        return other // self

    @always_inline("nodebug")
    fn __rmod__(self: Self, other: Self) -> Self:
        return other % self

    @always_inline("nodebug")
    fn __radd__(self: Self, other: SIMD[type, 1]) -> Self:
        return self + other

    @always_inline("nodebug")
    fn __rsub__(self: Self, other: SIMD[type, 1]) -> Self:
        return self - other

    @always_inline("nodebug")
    fn __rmul__(self: Self, other: SIMD[type, 1]) -> Self:
        return self * other

    @always_inline("nodebug")
    fn __rtruediv__(self: Self, other: SIMD[type, 1]) -> Self:
        return self / other

    @always_inline("nodebug")
    fn __rfloordiv__(self: Self, other: SIMD[type, 1]) -> Self:
        return self // other

    @always_inline("nodebug")
    fn __rmod__(self: Self, other: SIMD[type, 1]) -> Self:
        return self % other

    @always_inline("nodebug")
    fn __neg__(self: Self) -> Self:
        return self * (Scalar[type](1)).__neg__()

    @always_inline("nodebug")
    fn __pos__(self: Self) -> Self:
        return self * (Scalar[type](1)).__pos__()

    @always_inline("nodebug")
    fn __floor__(self: Self) -> Self:
        alias func: fn[Type: DType, nelts: Int] (SIMD[Type, nelts]) -> SIMD[
            Type, nelts
        ] = SIMD.__floor__
        var result = Self.Operation.tensor_operation[func](self.tensor)
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __ceil__(self: Self) -> Self:
        alias func: fn[Type: DType, nelts: Int] (SIMD[Type, nelts]) -> SIMD[
            Type, nelts
        ] = SIMD.__ceil__
        var result = Self.Operation.tensor_operation[func](self.tensor)
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn __trunc__(self: Self) -> Self:
        alias func: fn[Type: DType, nelts: Int] (SIMD[Type, nelts]) -> SIMD[
            Type, nelts
        ] = SIMD.__trunc__
        var result = Self.Operation.tensor_operation[func](self.tensor)
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline
    fn __abs__(self) -> Self:
        alias func: fn[Type: DType, nelts: Int] (SIMD[Type, nelts]) -> SIMD[
            Type, nelts
        ] = SIMD.__abs__
        var result = Self.Operation.tensor_operation[func](self.tensor)
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline
    fn __matmul__(self: Self, other: Self) -> Self:
        """
        Implements matrix multiplication for Tensor.
        The operation is defined as self @ other.
        """
        if self.rank() <= 2 and other.rank() <= 2:
            return matmul(self, other)
        else:
            return bmm(self, other)

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
        var result = Self.Operation.tensor_operation[func](self.tensor)
        var requires_grad = self.requires_grad
        return Tensor(
            result,
            requires_grad=requires_grad,
        )

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
        var result = Self.Operation.tensor_operation[func](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad or other.requires_grad
        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn pow(inout self: Self, pow: Self):
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

    @always_inline("nodebug")
    fn div(self: Self, other: Tensor[type]) -> Self:
        return self.__truediv__(other)

    @always_inline("nodebug")
    fn bmm(self: Self, other: Self) -> Self:
        return bmm(self, other)

    @always_inline("nodebug")
    fn fusedbmm[
        fuse_operation: fn[T: DType, nelts: Int] (SIMD[T, nelts]) -> SIMD[
            T, nelts
        ]
    ](self: Self, other: Self) -> Self:
        return fusedbmm[type, fuse_operation](self, other)

    @always_inline("nodebug")
    fn sum(self: Self) -> Self:
        """
        Compute the sum of all elements in the tensor.

        Returns:
            The sum of all elements in the tensor as a scalar value.
        """
        var result = Tensor[type](shape(1))
        alias simd_width: Int = simdwidthof[type]()

        @parameter
        fn vectorize_sum[simd_width: Int](idx: Int) -> None:
            var simd_data = self.load[simd_width](idx)
            result += simd_data.reduce_add()

        vectorize[vectorize_sum, simd_width](self.num_elements())
        return result

    @always_inline("nodebug")
    fn max(self: Self) -> Scalar[type]:
        """
        Find the maximum value in the tensor.

        Returns:
            The maximum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]()

        @parameter
        fn closure[nelts: Int](i: Int):
            result = max(result, self.load[nelts](i).reduce_max())

        vectorize[closure, nelts](
            self.num_elements() - (self.num_elements() % nelts)
        )
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result = max(result, self.load(i))

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
        fn closure[nelts: Int](i: Int):
            result = min(result, self.load[nelts](i).reduce_min())

        vectorize[closure, nelts](
            self.num_elements() - (self.num_elements() % nelts)
        )
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result = min(result, self.load(i))

        return result

    @always_inline("nodebug")
    fn mean(self: Self) -> Self:
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
        fn closure[nelts: Int](i: Int):
            result *= self.load[nelts](i).reduce_mul()

        vectorize[closure, nelts](
            self.num_elements() - (self.num_elements() % nelts)
        )
        for i in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            result *= self.load(i)
        return result

    @always_inline("nodebug")
    fn relu(self: Self) -> Self:
        return relu[type](self)

    @always_inline("nodebug")
    fn tanh(self: Self) -> Self:
        return tanh[type](self)

    @always_inline("nodebug")
    fn sigmoid(self: Self) -> Self:
        return sigmoid[type](self)

    @always_inline("nodebug")
    fn acos(self: Self) -> Self:
        return self.apply[math.acos]()

    @always_inline("nodebug")
    fn acosh(self: Self) -> Self:
        return self.apply[math.acosh]()

    @always_inline("nodebug")
    fn cos(self: Self) -> Self:
        return self.apply[cos]()

    @always_inline("nodebug")
    fn sinh(self: Self) -> Self:
        return self.apply[sinh]()

    @always_inline("nodebug")
    fn cosh(self: Self) -> Self:
        return self.apply[cosh]()

    @always_inline("nodebug")
    fn exp(self: Self) -> Self:
        return self.apply[math.exp]()

    @always_inline("nodebug")
    fn exp2(self: Self) -> Self:
        return self.apply[math.exp2]()

    @always_inline("nodebug")
    fn erf(self: Self) -> Self:
        return self.apply[erf]()

    @always_inline("nodebug")
    fn erfc(inout self: Self) -> Self:
        return self.apply[erfc]()

    @always_inline("nodebug")
    fn sin(self: Self) -> Self:
        return self.apply[sin]()

    @always_inline("nodebug")
    fn tan(self: Self) -> Self:
        return self.apply[math.tan]()

    @always_inline("nodebug")
    fn atan(self: Self) -> Self:
        return self.apply[math.atan]()

    @always_inline("nodebug")
    fn atanh(self: Self) -> Self:
        return self.apply[math.atanh]()

    @always_inline("nodebug")
    fn atan2(self: Self, other: Self) -> Self:
        return self.apply[math.atan2](other)

    @always_inline("nodebug")
    fn hypot(self: Self, other: Self) -> Self:
        return self.apply[math.hypot](other)

    @always_inline("nodebug")
    fn j0(self: Self) -> Self:
        return self.apply[j0]()

    @always_inline("nodebug")
    fn j1(self: Self) -> Self:
        return self.apply[math.j1]()

    @always_inline("nodebug")
    fn log1p(self: Self) -> Self:
        return self.apply[math.log1p]()

    @always_inline("nodebug")
    fn log(self: Self) -> Self:
        return self.apply[math.log]()

    @always_inline("nodebug")
    fn log10(self: Self) -> Self:
        return self.apply[math.log10]()

    @always_inline("nodebug")
    fn log2(self: Self) -> Self:
        return self.apply[math.log2]()

    @always_inline("nodebug")
    fn logb(self: Self) -> Self:
        return self.apply[math.logb]()

    @always_inline("nodebug")
    fn lgamma(self: Self) -> Self:
        return self.apply[math.lgamma]()

    @always_inline("nodebug")
    fn sclab(self: Self, other: Self) -> Self:
        return self.apply[math.scalb](other)

    @always_inline("nodebug")
    fn gamma(self: Self) -> Self:
        return self.apply[math.gamma]()

    @always_inline("nodebug")
    fn y0(self: Self) -> Self:
        return self.apply[math.y0]()

    @always_inline("nodebug")
    fn y1(self: Self) -> Self:
        return self.apply[math.y1]()

    @always_inline("nodebug")
    fn ulp(self: Self) -> Self:
        return self.apply[math.ulp]()

    @always_inline("nodebug")
    fn cbrt(self: Self) -> Self:
        return self.apply[math.cbrt]()

    @always_inline("nodebug")
    fn rsqrt(self: Self) -> Self:
        return self.apply[math.rsqrt]()

    @always_inline("nodebug")
    fn roundeven(self: Self) -> Self:
        return self.apply[SIMD.roundeven]()
    
    @always_inline("nodebug")
    fn nextafter(self: Self, other: Self) -> Self:
        var result = Self.Operation.tensor_operation[math.nextafter](
            self.tensor, other.tensor
        )
        var requires_grad = self.requires_grad

        return Tensor(
            result,
            requires_grad=requires_grad,
        )

    @always_inline("nodebug")
    fn list(self) -> List[Scalar[type]]:
        var result = List[Scalar[type]]()
        for i in range(self.num_elements()):
            result.append(self.load(i))
        return result^

    @always_inline("nodebug")
    fn zeros(inout self: Self):
        memset_zero(self.tensor.data, self.num_elements())

    @always_inline("nodebug")
    fn require_grad(inout self, requires_grad: Bool = True):
        self = Self(self.tensor, requires_grad=requires_grad)

    @always_inline("nodebug")
    fn ones(inout self: Self):
        memset[type](self.tensor.data, 1, self.num_elements())

    @always_inline("nodebug")
    fn rand(inout self):
        rand_n[type](self.tensor.data, self.num_elements())

    @always_inline("nodebug")
    fn rand(inout self, seed: Int):
        rand_n[type](self.tensor.data, self.num_elements(), seed)

    @always_inline("nodebug")
    fn random(self) -> Self:
        var rand = self
        rand.rand()
        return rand^

    @always_inline("nodebug")
    fn random(self, seed: Int) -> Self:
        var rand = self
        rand.rand(seed)
        return rand

    @always_inline("nodebug")
    fn fill(self: Self, value: Scalar[type]) -> Self:
        """Fill the tensor with a specified value."""
        var result = DTypePointer[type]().alloc(self.num_elements())
        for i in range(0, self.num_elements()):
            result[i] = result[i].splat(value)
        return Tensor[type](TensorType[type](self.shapes(), result, self.device()), self.requires_grad)

    @always_inline("nodebug")
    fn ifill(inout self: Self, value: Scalar[type]):
        """Fill the tensor with a specified value."""
        self = self.fill(value)

    @always_inline("nodebug")
    fn transposed(inout self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:
        if dim1 >= self.rank() or dim2 >= self.rank():
            handle_issue("dim1 and dim2 must be within the range of the tensor's rank")

        var tshape = self.shapes()
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        var ttensor = Tensor[type](tshape)

        for index in range(self.num_elements()):
            var _indices = self.tensor.shape.indices(index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]
            ttensor.store(tindices, self.load(index))

        return ttensor^

    @always_inline("nodebug")
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var ttensor = self.transposed(dim1, dim2)
        self = ttensor

    @always_inline("nodebug")
    fn reshape(self: Self, other: shape) -> Self:
        """Reshape the tensor to the new dimensions and returns the reshaped tensor.
        """
        if self.tensor.shape.num_elements != other.num_elements:
            handle_issue("Cannot reshape tensor.")

        var data = Tensor[type](other)

        @parameter
        fn closure[nelts: Int](idx: Int):
            var old_indices = self.tensor.shape.indices(idx)
            var new_indices = other.indices(idx)
            data.store(new_indices, self.load(old_indices))

        vectorize[closure, 1](self.num_elements())
        return Self(other, data.tensor.data)

    fn broadcast_to(inout self: Self, other: shape):
        var Other = Tensor[type](other)
        Other.zeros()
        var result = Self.Operation.tensor_operation[SIMD.__add__](
            self.tensor, Other.tensor
        )
        var requires_grad = self.requires_grad

        self = Tensor(
            result,
            requires_grad=requires_grad,
        )

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
        return self.tensor.shape.rank

    @always_inline("nodebug")
    fn shapes(self: Self) -> shape:
        return self.tensor.shape

    @always_inline("nodebug")
    fn data(self: Self) -> DTypePointer[type]:
        return self.tensor.data

    @always_inline("nodebug")
    fn data(self, shapes: shape, offset: Int) -> DTypePointer[type]:
        var num_elements = shapes.num_elements
        var new_data = DTypePointer[type]().alloc(num_elements)
        memcpy(new_data, self.tensor.data + (offset), num_elements)
        return new_data

    @always_inline("nodebug")
    fn unsafe_ptr(self) -> UnsafePointer[Scalar[type]]:
        var data = UnsafePointer[Scalar[type]]().alloc(self.num_elements())
        for i in range(self.num_elements()):
            data[i] = self.tensor.data[i]
        return data

    @always_inline("nodebug")
    fn dtype(self: Self) -> String:
        return str(type)

    # TODO: Add support for GPU
    @always_inline("nodebug")
    fn device(self: Self) -> StringLiteral:
        return self.tensor.device

    @always_inline("nodebug")
    fn num_elements(self: Self) -> Int:
        return self.tensor.shape.numofelements()

    @always_inline("nodebug")
    fn astype[dest: DType](self: Self) -> Tensor[dest]:
        var casted = Tensor[dest](self.tensor.shape)
        alias nelts = simdwidthof[dest]() * 2

        @parameter
        fn cast_single_element[nelts: Int](index: Int):
            casted.tensor.store[nelts](
                index, self.tensor.load[nelts](index).cast[dest]()
            )

        vectorize[cast_single_element, nelts](
            self.num_elements() - (self.num_elements() % nelts),
        )
        for index in range(
            self.num_elements() - (self.num_elements() % nelts),
            self.num_elements(),
        ):
            casted.store(index, self.load(index).cast[dest]())
        return casted^

    @always_inline("nodebug")
    fn cast[dest: DType](self: Self) -> Tensor[dest]:
        return self.astype[dest]()

    @always_inline("nodebug")
    fn num_bytes(self: Self) -> Int:
        return sizeof[type]() * self.num_elements()

    @always_inline("nodebug")
    fn itemsize(self: Self) -> Int:
        return sizeof[type]()

    @always_inline("nodebug")
    fn serialize(self) -> Serialize:
        return Serialize(self)

    @always_inline("nodebug")
    @staticmethod
    fn deserialize(data: Serialize) -> Tensor[type]:
        return Serialize.totensor[type](data)


fn tensor[
    dtype: DType = DType.float32
](Shape: List[Int], rand: Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor^


fn tensor[
    dtype: DType = DType.float32
](*Shape: Int, rand: Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor^


fn ones[
    dtype: DType = DType.float32
](Shape: List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](1))
    return tensor^


fn ones[
    dtype: DType = DType.float32
](*Shape: Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](1))
    return tensor^


fn zeros[
    dtype: DType = DType.float32
](Shape: List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](0))
    return tensor^


fn zeros[
    dtype: DType = DType.float32
](*Shape: Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, Scalar[dtype](0))
    return tensor^


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
    return tensor^


fn fill[
    dtype: DType = DType.float32
](Shape: List[Int], val: Scalar[dtype]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor, val)
    return tensor^


fn rand[
    dtype: DType = DType.float32
](Shape: List[Int], seed: Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand()
    tensor.rand()
    return tensor^


fn rand[
    dtype: DType = DType.float32
](*Shape: Int, seed: Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand()
    tensor.rand()
    return tensor^


fn empty[dtype: DType = DType.float32](*Shape: Int) -> Tensor[dtype]:
    return Tensor[dtype](Shape)


fn empty[dtype: DType = DType.float32](Shape: List[Int]) -> Tensor[dtype]:
    return Tensor[dtype](Shape)


@always_inline("nodebug")
fn arange[dtype: DType](start: Int, end: Int = 0, step: Int = 1) -> Tensor[dtype]:
    """
    Returns a tensor with values from start to end with specified step size.

    Args:
        start: The start value of the sequence.
        end: The end value of the sequence.
        step: The step size between consecutive values. Default is 1.

    Returns:
        A tensor containing the values from start to end with the specified step size.
    """
    var num: Int = int((end - start) / step)
    var result: Tensor[dtype] = Tensor[dtype](shape(num))
    for idx in range(num):
        result[idx] = start + step * idx
    return result