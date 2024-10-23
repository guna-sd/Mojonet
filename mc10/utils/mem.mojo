@value
@register_passable("trivial")
struct DataPointer(
    ImplicitlyBoolable,
    CollectionElement,
    CollectionElementNew,
    Stringable,
    Formattable,
    Intable,
    Comparable,
):
    """The DataPointer struct essentially wraps a raw pointer (UnsafePointer[NoneType]) that can point to any type of Scalar data.
    It links the raw pointer to a specific device (Device), which is useful when managing data across different hardware (e.g., CPU vs GPU).
    """
    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    alias __ptr_type = UnsafePointer[NoneType]
    alias Null = Self.__ptr_type()

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var ptr: Self.__ptr_type
    """The underlying pointer."""

    var device: Device
    """The device on which the pointer is located (e.g., CPU or GPU)."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self):
        self.device = Device.CPU
        self.ptr = Self.Null
    
    fn __init__(inout self, device: Device = Device.CPU):
        self.device = device
        self.ptr = Self.Null
    
    fn __init__(inout self, owned ptr: Self.__ptr_type, owned device: Device):
        self.ptr = ptr
        self.device = device

    fn __init__(inout self: DataPointer, /, *, other: DataPointer) -> None:
        self.ptr = other.ptr
        self.device = other.device

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn alloc(inout self, size: Int):
        self.ptr = __malloc[NoneType](size)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[T : DType](ref[_]self) -> ref[self] Scalar[T]:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """
        return self.ptr.bitcast[T]()[]

    @always_inline
    fn __getitem__[T : DType](ref[_]self, offset: Int) -> ref[self] Scalar[T]:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """
        return self.ptr.bitcast[T]()[offset]

    @always_inline("nodebug")
    fn offset[T: IntLike](self, idx: T) -> DataPointer:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed DataPointer.
        """
        return Self(self.ptr.offset(idx), self.device)

    @always_inline("nodebug")
    fn __add__[T: IntLike](self, rhs: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new DataPointer shifted by the offset.
        """
        return self.offset(rhs)

    @always_inline("nodebug")
    fn __sub__[T: Intable](self, rhs: T) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new DataPointer shifted by the offset.
        """
        return self.offset(-int(rhs))

    @always_inline("nodebug")
    fn __iadd__[T: IntLike](inout self, rhs: T):
        """Shifts the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__[T: Intable](inout self, rhs: T):
        """Shifts back the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.ptr == rhs.ptr

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.ptr != rhs.ptr

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.ptr < rhs.ptr

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return int(self) <= int(rhs)

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and False otherwise.
        """
        return int(self) > int(rhs)

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and False otherwise.
        """
        return int(self) >= int(rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return int(self) != 0

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
        return int(self.ptr)

    @no_inline
    fn __str__(self) -> String:
        """Format this pointer as a hexadecimal string.

        Returns:
            A String containing the hexadecimal representation of the memory location
            destination of this pointer.
        """
        return str(self.ptr)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this pointer address to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write(str(self))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn free(self):
        """Free the memory referenced by the pointer."""
        __free(self.ptr)
    
    @always_inline("nodebug")
    fn __set[Type: DType](inout self, owned value: Scalar[Type]):
        self.ptr.bitcast[Type]().init_pointee_move(value)

    @always_inline("nodebug")    
    fn __get[Type: DType](self) -> Scalar[Type]:
        return self.ptr.bitcast[Type]()[]

    @always_inline("nodebug")
    fn __set[Type: DType](inout self, offset: Int, owned value: Scalar[Type]):
        (self.ptr.bitcast[Type]() + offset).destroy_pointee()
        (self.ptr.bitcast[Type]() + offset).init_pointee_move(value)

    @always_inline("nodebug")    
    fn __get[Type: DType](self, offset: Int) -> Scalar[Type]:
        return (self.ptr.bitcast[Type]() + offset)[]