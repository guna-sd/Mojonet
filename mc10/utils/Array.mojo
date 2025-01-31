from collections._index_normalization import normalize_index
from memory import UnsafePointer, bitcast
from mc10.__mlir import (
    _mlirtype_is_eq,
    _type_is_eq,
    is_trivial,
    implements,
    int,
    long,
    float,
    double,
)
from utils import StaticTuple
from os import abort

alias ByteArray = Array[Byte, *_]
alias IntArray = Array[int, *_]
alias LongArray = Array[long, *_]
alias FloatArray = Array[float, *_]
alias DoubleArray = Array[double, *_]


@always_inline
fn _set_array_elem_move[
    type: CollectionElement,
    capacity: Int,
](
    index: Int,
    owned element: type,
    ref array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer.address_of(array).address, index.value
    )
    UnsafePointer(ptr).init_pointee_move(element)

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(element))


@always_inline
fn _set_array_elem_copy[
    type: CollectionElement,
    capacity: Int,
](
    index: Int,
    owned element: type,
    ref array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer.address_of(array).address, index.value
    )
    UnsafePointer(ptr).init_pointee_copy(element)


@always_inline
fn _create_array[
    type: CollectionElement, capacity: Int
](owned storage: VariadicListMem[type]) -> Array[type, capacity]:
    array = Array[type, capacity]()
    array.size = len(storage)

    @parameter
    for idx in range(capacity):
        _set_array_elem_move[type, capacity](idx, storage[idx], array.storage)

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(storage))

    return array


fn _init_array_default[
    capacity: Int, type: CollectionElement
](default: type) -> __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`]:
    var array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(array))

    @parameter
    for idx in range(capacity):
        _set_array_elem_copy[type, capacity](idx, default, array)
    return array


fn _array_construction_checks[size: Int]():
    constrained[size > 0, "number of elements in `Array` must be > 0"]()


@value
@register_passable("trivial")
struct Array[Type: CollectionElement, capacity: Int]:
    alias __array_type = __mlir_type[
        `!pop.array<`, Self.capacity.value, `, `, Self.Type, `>`
    ]

    var storage: Self.__array_type
    var size: Int

    fn __init__(out self):
        """
        Unsafe initialization, provide default if primitive eg. `Array[Int, 32](0)`.
        """
        _array_construction_checks[capacity]()
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )
        self.size = 0

    @always_inline
    @implicit
    fn __init__(out self, default: Type):
        self = Self()
        self.storage = _init_array_default[capacity, Type](default)

    @always_inline
    fn __init__(out self, owned *elements: Self.Type):
        self = Self(storage=elements^)

    @always_inline
    fn __init__(out self, owned storage: VariadicListMem[Self.Type, _]):
        debug_assert(
            len(storage) <= capacity,
            "number of elements in storage is too large",
        )
        self = _create_array[Type, capacity](storage^)

    fn __init__(out self, list: List[Type, _]):
        debug_assert(
            capacity == list.capacity,
            "mismatch in the number of elements in the list",
        )
        self = Self()
        self.size = list.size

        @parameter
        for i in range(capacity):
            self[i] = list[i]

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        var copy = Self()
        copy.size = self.size

        @parameter
        for idx in range(capacity):
            ptr = copy.unsafe_ptr() + idx
            ptr.init_pointee_copy(self[idx])

        return copy

    @always_inline
    fn __getitem__(ref self, index: Int) -> ref [self.storage] Self.Type:
        return self.unsafe_get(index)

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the array. This is a known constant value."""
        return self.size

    @always_inline
    fn __bool__(self) -> Bool:
        return len(self) > 0

    fn __contains__[
        T: EqualityComparableCollectionElement, //
    ](self: Array[T, *_], value: T) -> Bool:
        @parameter
        for i in range(capacity):
            if self[i] == value:
                return True
        return False

    @always_inline
    fn unsafe_get(ref self, index: Int) -> ref [self.storage] Self.Type:
        debug_assert(
            -self.size <= index < self.size,
            " Array.unsafe_get() index out of bounds: ",
            index,
            " should be less than: ",
            capacity,
        )
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(self.storage).address,
            index.value,
        )
        return UnsafePointer(ptr)[]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self.Type]:
        """Get an `UnsafePointer` to the underlying array.

        Returns:
            An `UnsafePointer` to the underlying array.
        """
        return UnsafePointer.address_of(self.storage).bitcast[Self.Type]()

    fn list(read self) -> List[Self.Type, True]:
        var list = List[Self.Type, True](capacity=capacity)

        @parameter
        for i in range(capacity):
            list[i] = self[i]
        return list^

    fn toString[
        T: RepresentableCollectionElement
    ](read self: Array[T, *_]) -> String:
        var string = String()
        string.write("[")

        for i in range(capacity):
            if i >= self.size:
                string.write("null")
            else:
                string.write(repr(self[i]))
            if i < capacity - 1:
                string.write(", ")
        string.write("]")
        return string

    @no_inline
    fn __str__[
        T: RepresentableCollectionElement, //
    ](self: Array[T, *_]) -> String:
        return self.toString()

    @no_inline
    fn __repr__[
        T: RepresentableCollectionElement, //
    ](self: Array[T, *_]) -> String:
        return self.__str__()


struct Arrays:
    @staticmethod
    fn strfromArray(read array: Array) -> String:
        constrained[
            _type_is_eq[
                Array[UInt8, array.capacity].__array_type, array.__array_type
            ](),
            "Type must be a UInt8 to write a string from array",
        ]()
        var p = array.list()
        var ptr = p.steal_data()
        var str = String(
            ptr=rebind[UnsafePointer[UInt8]](ptr), length=array.capacity
        )
        return str^

    @staticmethod
    fn min(read array: Array[Int]) -> Int:
        min = Int.MAX
        for i in array.list():
            if i[] < min:
                min = i[]
        return min
