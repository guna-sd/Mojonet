from memory import UnsafePointer
from mc10.types.Array import (
    _array_construction_checks,
    _create_array,
    indexable,
    asserts,
)

alias ByteArray = DTypeArray[DType.uint8, *_]
alias IntArray = DTypeArray[DType.int32, *_]
alias LongArray = DTypeArray[DType.int64, *_]
alias FloatArray = DTypeArray[DType.float32, *_]
alias DoubleArray = DTypeArray[DType.float64, *_]


@value
@register_passable("trivial")
struct DTypeArray[dtype: DType, size: Int]:
    alias __array_type = __mlir_type[
        `!pop.array<`, size.value, `,`, Scalar[dtype], `>`
    ]

    var array: Self.__array_type

    fn __init__(out self):
        _array_construction_checks[size]()
        constrained[
            dtype is not DType.invalid, "dtype cannot be DType.invalid"
        ]()
        var zero: Scalar[dtype] = __mlir_op.`pop.cast`[
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`]
        ](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<index>`],
                value = __mlir_attr[`#pop.simd<0> : !pop.scalar<index>`],
            ]()
        )
        self.array = __mlir_op.`pop.array.repeat`[_type = Self.__array_type](
            zero
        )

    @always_inline
    fn __init__(out self, *, unsafe_uninitialized: Bool):
        """Constructs a DTypeArray with uninitialized memory.
        Note that this is highly unsafe and should be used with caution.

        Args:
            unsafe_uninitialized: A boolean to indicate if the array
                should be initialized. Always set to `True`
                (it's not actually used inside the constructor).
        """
        self.array = __mlir_op.`kgen.param.constant`[
            _type = Self.__array_type,
            value = __mlir_attr[`#kgen.unknown : `, Self.__array_type],
        ]()

    fn __init__(out self, fill: Scalar[dtype]):
        _array_construction_checks[size]()
        constrained[
            dtype is not DType.invalid, "dtype cannot be DType.invalid"
        ]()

        self.array = __mlir_op.`pop.array.repeat`[_type = Self.__array_type](
            fill
        )

    @always_inline
    @implicit
    fn __init__(out self, *elements: Scalar[Self.dtype]):
        self = Self(storage=elements)

    @always_inline
    @implicit
    fn __init__(out self, storage: VariadicList[Scalar[Self.dtype]]):
        asserts(
            len(storage) <= size,
            "number of elements in storage is too large",
        )
        self.array = _create_array[Scalar[Self.dtype], size](storage)

    @always_inline
    @implicit
    fn __init__(out self, array: Self.__array_type):
        self.array = array

    fn __init__(out self, *, other: Self):
        self.array = other.array

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return size

    @always_inline("nodebug")
    fn __getitem__[index: Int](self) -> Scalar[dtype]:
        constrained[index < size]()
        return __mlir_op.`pop.array.get`[
            _type = Scalar[dtype],
            index = index.value,
        ](self.array)

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(self.array).address,
            indexable["Array"](idx, self).value,
        )
        return UnsafePointer(ptr)[]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, val: Scalar[dtype]):
        var tmp = self
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(tmp.array).address,
            indexable["Array"](idx, self).value,
        )
        UnsafePointer(ptr)[] = val
        self = tmp

    @always_inline("nodebug")
    fn __contains__(self, value: Scalar[dtype]) -> Bool:
        @parameter
        for i in range(size):
            if self[i] == value:
                return True
        return False

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut buffer: W):
        buffer.write("[")

        @parameter
        for i in range(size):
            buffer.write(self[i])
            if i < size - 1:
                buffer.write(", ")
        buffer.write("]")
        return
