from memory import UnsafePointer
from mc10.utils.memutils import (
    __malloc,
    __free,
    _default_alignment,
    PrefetchOptions,
    PrefetchLocality,
    PrefetchCache,
    PrefetchRW,
    prefetch,
)


@value
@register_passable("trivial")
struct DataPointer[
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = 1,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
](
    ImplicitlyBoolable,
    CollectionElement,
    CollectionElementNew,
    Stringable,
    Writable,
    Intable,
    Comparable,
):
    """The DataPointer struct essentially wraps a raw pointer (UnsafePointer[NoneType]) that can point to any type of Scalar data.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    alias __ptr_type = UnsafePointer[
        type=NoneType,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]
    alias Null = Self.__ptr_type()

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self.__ptr_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        self.address = Self.Null

    @always_inline
    @implicit
    fn __init__(out self, __null_type: None):
        self = Self()

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, __ptr_type: Self.__ptr_type):
        self.address = __ptr_type

    @always_inline
    @implicit
    fn __init__(out self, other: DataPointer[address_space=address_space, **_]):
        self.address = other.address

    @always_inline
    @implicit
    fn __init__(
        out self, other: UnsafePointer[address_space=address_space, **_]
    ):
        self.address = other.bitcast[NoneType]()

    @always_inline
    fn copy(self) -> Self:
        return DataPointer(self.address)

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    fn address_of[
        type: AnyType
    ](
        ref [address_space]arg: type,
        out result: DataPointer[
            address_space=address_space,
            alignment=alignment,
            mut = Origin(__origin_of(arg)).is_mutable,
            origin = __origin_of(arg),
        ],
    ):
        """Gets the address of the argument.

        Args:
            arg: The value to get the address of.

        Returns:
            An Pointer which contains the address of the argument.
        """
        var addr: UnsafePointer[
            type,
            address_space=address_space,
            alignment=alignment,
            mut = Origin(__origin_of(arg)).is_mutable,
            origin = __origin_of(arg),
        ]
        addr = __type_of(addr)(
            __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(arg))
        )
        return __type_of(result)(addr.bitcast[NoneType]())

    @always_inline
    @staticmethod
    fn alloc(size: Int) -> DataPointer[address_space=address_space, **_]:
        ptr = (
            UnsafePointer[
                NoneType,
                address_space=address_space,
                alignment=alignment,
                mut=mut,
                origin=origin,
            ]()
            .alloc(size)
            .address_space_cast[address_space]()
            .origin_cast[mut, origin]()
        )
        return Self(ptr)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[T: DType](self) -> ref [origin, address_space] Scalar[T]:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """
        return self.address.bitcast[Scalar[T]]()[]

    @always_inline
    fn __getitem__[
        T: DType
    ](self, offset: Int) -> ref [origin, address_space] Scalar[T]:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """
        return (self + offset).address.bitcast[Scalar[T]]()[]

    @always_inline("nodebug")
    fn offset(self, idx: Int) -> DataPointer[address_space, **_]:
        """Returns a new pointer shifted by the specified offset.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed DataPointer.
        """
        return Self(self.address.offset(idx))

    @always_inline("nodebug")
    fn __add__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new DataPointer shifted by the offset.
        """
        return self.address + rhs

    @always_inline("nodebug")
    fn __sub__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new DataPointer shifted by the offset.
        """
        return self.address - rhs

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Int):
        """Shifts the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Int):
        """Shifts back the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.address == rhs.address

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.address != rhs.address

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.address < rhs.address

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and False otherwise.
        """
        return Int(self) >= Int(rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return Int(self) != 0

    @always_inline
    fn __as_bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return self.__bool__()

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(self.address)

    @always_inline
    fn __as_int__(self) -> Int:
        return Int(self.address)

    @no_inline
    fn __str__(self) -> String:
        """Format this pointer as a hexadecimal string.

        Returns:
            A String containing the hexadecimal representation of the memory location
            destination of this pointer.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this pointer address to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write(self.address)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn free(self: DataPointer[address_space = AddressSpace.GENERIC, **_]):
        """Free the memory referenced by the pointer."""
        self.address.free()

    @always_inline("nodebug")
    fn store(mut self, owned value: Scalar):
        self.address.bitcast[__type_of(value)]().store(value)

    @always_inline("nodebug")
    fn store(mut self, offset: Int, owned value: Scalar):
        self.address.bitcast[__type_of(value)]().store(offset, value)

    @always_inline("nodebug")
    fn store[
        dtype: DType, width: Int
    ](mut self, owned value: SIMD[dtype, width]):
        self.address.bitcast[Scalar[dtype]]().store(value)

    @always_inline("nodebug")
    fn store[
        dtype: DType, width: Int
    ](mut self, offset: Int, owned value: SIMD[dtype, width]):
        self.address.bitcast[Scalar[dtype]]().store(offset, value)

    @always_inline("nodebug")
    fn load[Type: DType](self) -> Scalar[Type]:
        return self.address.bitcast[Scalar[Type]]().load()

    @always_inline("nodebug")
    fn load[Type: DType, width: Int](self) -> SIMD[Type, width]:
        return self.address.bitcast[Scalar[Type]]().load[width=width]()

    @always_inline("nodebug")
    fn load[Type: DType](self, offset: Int) -> Scalar[Type]:
        return self.address.bitcast[Scalar[Type]]().load(offset)

    @always_inline("nodebug")
    fn load[Type: DType, width: Int](self, offset: Int) -> SIMD[Type, width]:
        return self.address.bitcast[Scalar[Type]]().load[width=width](offset)

    @always_inline
    fn prefetch[
        dtype: DType, //, params: PrefetchOptions = PrefetchOptions()
    ](self):
        prefetch[params](self.address.bitcast[Scalar[dtype]]())
