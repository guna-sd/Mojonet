from memory import UnsafePointer, bitcast
from mc10.__mlir import _type_is_eq
from mc10.utils.index import indexable
from mc10.utils.debuggable import asserts


@always_inline
fn _set_array_elem_move[
    type: Copyable & Movable,
    capacity: Int,
](
    index: Int,
    owned element: type,
    ref array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer(to=array).address, index.value
    )
    UnsafePointer(ptr).init_pointee_move(element)

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(element))


@always_inline
fn _set_array_elem_copy[
    type: Copyable & Movable,
    capacity: Int,
](
    index: Int,
    element: type,
    ref array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`],
):
    ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer(to=array).address, index.value
    )
    UnsafePointer(ptr).init_pointee_copy(element)


@always_inline
fn _create_array[
    type: Copyable & Movable, capacity: Int
](owned storage: VariadicListMem[type]) -> Array[type, capacity]:
    array = Array[type, capacity]()
    array.size = len(storage)

    for idx in range(len(storage)):
        _set_array_elem_move[type, capacity](idx, storage[idx], array.storage)

    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(storage))

    return array


@always_inline
fn _create_array[
    type: AnyTrivialRegType, size: Int
](storage: VariadicList[type]) -> __mlir_type[
    `!pop.array<`, size.value, `, `, type, `>`
]:
    if len(storage) == 1:
        return __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[`!pop.array<`, size.value, `, `, type, `>`]
        ](storage[0])

    asserts(size == len(storage), "mismatch in the number of elements")

    var array: __mlir_type[`!pop.array<`, size.value, `, `, type, `>`]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(array))

    @parameter
    for idx in range(size):
        _set_array_elem_copy[type, size](idx, storage[idx], array)

    return array


fn _init_array_default[
    capacity: Int, type: Copyable & Movable
](default: type) -> __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`]:
    var array: __mlir_type[`!pop.array<`, capacity.value, `, `, type, `>`]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(array))

    @parameter
    for idx in range(capacity):
        _set_array_elem_copy[type, capacity](idx, default, array)
    return array


fn _array_construction_checks[size: Int]():
    constrained[size > 0, "number of elements in `Array` must be > 0"]()


@fieldwise_init
@register_passable("trivial")
struct Array[Type: Copyable & Movable, capacity: Int](Sized):
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
        asserts(
            len(storage) <= capacity,
            "number of elements in storage is too large",
        )
        self = _create_array[Type, capacity](storage^)

    @always_inline
    @implicit
    fn __init__(out self, list: List[Type, _]):
        asserts(
            capacity == list.capacity,
            "mismatch in the number of elements in the list",
        )
        self = Self()
        self.size = list._len

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
        T: Copyable & Movable & EqualityComparable, //
    ](self: Array[T, *_], value: T) -> Bool:
        @parameter
        for i in range(capacity):
            if self[i] == value:
                return True
        return False

    @always_inline
    fn unsafe_get(ref self, index: Int) -> ref [self.storage] Self.Type:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self.storage).address,
            indexable["Array"](index, self).value,
        )
        return UnsafePointer(ptr)[]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self.Type]:
        """Get an `UnsafePointer` to the underlying array.

        Returns:
            An `UnsafePointer` to the underlying array.
        """
        return UnsafePointer(to=self.storage).bitcast[Self.Type]()

    fn list(read self) -> List[Self.Type, True]:
        var list = List[Self.Type, True](capacity=capacity)
        list._len = self.size

        @parameter
        for i in range(capacity):
            list[i] = self[i]
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
            if i[] < min:
                min = i[]
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
