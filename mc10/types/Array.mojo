from memory.unsafe_pointer import UnsafePointer
from sys.intrinsics import _type_is_eq
from builtin.debug_assert import debug_assert as asserts
from os import abort


@always_inline
fn indexable[
    Type: Sized, //, name: StringLiteral
](idx: Int, container: Type) -> Int:
    """
    Ensures `idx` is within bounds and returns the indexed value.

    Args:
        idx : The index to access.
        container : A sized container (array, list, etc.).

    Returns:
        Indexable in the container.
    """

    asserts(len(container) >= 0, "Container must have a valid size!")

    asserts(
        len(container) > 0,
        "Attempting to index into an empty ",
        name,
        " container with 0 elements!",
    )
    if idx < 0:
        print(String("Index must be non-negative! Got idx: ") + idx.__str__())
        abort()
    if idx >= len(container):
        print(
            String("Index out of bounds! Provided idx: ")
            + idx.__str__()
            + String(" is larger than container length: ")
            + len(container).__str__()
        )
        abort()

    return idx


@always_inline
fn _set_array_elem_move[
    type: Movable,
    capacity: Int,
](
    index: Int,
    var element: type,
    ref array: __mlir_type[
        `!pop.array<`, capacity._mlir_value, `, `, type, `>`
    ],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer(to=array).address, index._mlir_value
    )
    UnsafePointer(ptr).init_pointee_move(element^)

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(element))


@always_inline
fn _set_array_elem_copy[
    type: Copyable,
    capacity: Int,
](
    index: Int,
    element: type,
    ref array: __mlir_type[
        `!pop.array<`, capacity._mlir_value, `, `, type, `>`
    ],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer(to=array).address, index._mlir_value
    )
    UnsafePointer(ptr).init_pointee_copy(element)


@always_inline
fn _create_array[
    type: Copyable & Movable, capacity: Int
](var storage: VariadicListMem[type]) -> Array[type, capacity]:
    array = Array[type, capacity]()
    array.size = len(storage)

    for idx in range(len(storage)):
        _set_array_elem_move[type, capacity](
            idx, storage[idx].copy(), array.storage
        )

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(storage))

    return array


@always_inline
fn _create_array[
    type: AnyTrivialRegType, size: Int
](storage: VariadicList[type]) -> __mlir_type[
    `!pop.array<`, size._mlir_value, `, `, type, `>`
]:
    if len(storage) == 1:
        return __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[
                `!pop.array<`, size._mlir_value, `, `, type, `>`
            ]
        ](storage[0])

    asserts(size == len(storage), "mismatch in the number of elements")

    var array: __mlir_type[`!pop.array<`, size._mlir_value, `, `, type, `>`]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(array))

    @parameter
    for idx in range(size):
        _set_array_elem_copy[type, size](idx, storage[idx], array)

    return array


fn _init_array_default[
    size: Int, type: Copyable & Movable
](default: type) -> __mlir_type[
    `!pop.array<`, size._mlir_value, `, `, type, `>`
]:
    var array: __mlir_type[`!pop.array<`, size._mlir_value, `, `, type, `>`]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(array))

    @parameter
    for idx in range(size):
        _set_array_elem_copy[type, size](idx, default, array)
    return array


fn _array_construction_checks[size: Int]():
    constrained[size > 0, "number of elements in `Array` must be > 0"]()




# TODO: decide a much more cleaner way to have this... (High Priority)
# This plays an important role for various optimizations in the framework...

@register_passable("trivial")
struct Array[Type: Copyable & Movable, capacity: Int](Sized):
    alias __array_type = __mlir_type[
        `!pop.array<`, Self.capacity._mlir_value, `, `, Self.Type, `>`
    ]

    var storage: Self.__array_type
    var size: Int

    fn __init__(out self):
        """
        Unsafe initialization, provide default if primitive eg. `Array[Int, 32](0)`.
        """
        _array_construction_checks[Self.capacity]()
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )
        self.size = 0

    @always_inline
    @implicit
    fn __init__(out self, default: Self.Type):
        self = Self()
        self.storage = _init_array_default[Self.capacity, Self.Type](default)

    @always_inline
    fn __init__(out self, var *elements: Self.Type):
        self = Self(storage=elements^)

    @always_inline
    fn __init__(out self, var storage: VariadicListMem[Self.Type, _]):
        asserts(
            len(storage) <= Self.capacity,
            "number of elements in storage is too large",
        )
        self = _create_array[Self.Type, Self.capacity](storage^)

    @always_inline
    @implicit
    fn __init__(out self, list: List[Self.Type]):
        asserts(
            Self.capacity == list.capacity,
            "mismatch in the number of elements in the list",
        )
        self = Self()
        self.size = list._len

        @parameter
        for i in range(Self.capacity):
            self[i] = list[i].copy()

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        var copy = Self()
        copy.size = self.size

        @parameter
        for idx in range(Self.capacity):
            ptr = UnsafePointer(to=copy.storage).bitcast[Self.Type]() + idx
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
        T: Copyable & Movable & Equatable, //
    ](self: Array[T, *_], value: T) -> Bool:
        @parameter
        for i in range(Self.capacity):
            if self[i] == value:
                return True
        return False

    @always_inline
    fn unsafe_get(ref self, index: Int) -> ref [self.storage] Self.Type:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self.storage).address,
            indexable["Array"](index, self)._mlir_value,
        )
        return UnsafePointer(ptr)[]

    @always_inline
    fn unsafe_ptr[
        mut: Bool,
        origin: Origin[mut], //,
    ](self) -> UnsafePointer[Self.Type, origin]:
        """Get an `UnsafePointer` to the underlying array.

        Returns:
            An `UnsafePointer` to the underlying array.
        """
        return (
            UnsafePointer(to=self.storage)
            .bitcast[Self.Type]()
            .unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )

    fn list(read self) -> List[Self.Type]:
        var list = List[Self.Type](capacity=Self.capacity)
        list._len = self.size

        @parameter
        for i in range(Self.capacity):
            list[i] = self[i].copy()
        return list^


struct Arrays:
    @staticmethod
    fn strfromArray(read array: Array) -> String:
        constrained[
            _type_is_eq[
                Array[UInt8, array.capacity].__array_type, array.__array_type
            ](),
            "Type must be a UInt8 to write a string from array",
        ]()

        ## TODO: This is still a workaround for the fact that we don't have a proper way to convert an array to a string
        var p = array.list()
        var ptr = p.steal_data()
        var str = String()
        str._len_or_data = array.capacity
        str._ptr_or_data = ptr.bitcast[Byte]()
        return str^

    @staticmethod
    fn min(read array: Array[Int]) -> Int:
        min = Int.MAX
        for i in array.list():
            if i < min:
                min = i
        return min

    @staticmethod
    fn toString[
        T: Writable & Copyable & Movable
    ](read self: Array[T, *_]) -> String:
        var string = String()
        string.write("[")

        for i in range(self.capacity):
            if i >= self.size:
                string.write("null")
            else:
                string.write(self[i])
            if i < self.capacity - 1:
                string.write(", ")
        string.write("]")
        return string
