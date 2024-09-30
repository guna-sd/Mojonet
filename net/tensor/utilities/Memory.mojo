from memory.unsafe_pointer import is_power_of_two, _free
from memory.memory import memcpy, _malloc


@value
@register_passable("trivial")
struct DataPointer(
    Boolable, CollectionElement, Intable, Stringable, EqualityComparable
):
    alias _pointer_type = UnsafePointer[UInt8]
    var address: Self._pointer_type
    """The pointed-to address."""
    var device: Device
    """The device associated with."""

    @always_inline
    fn __init__(inout self):
        self.address = UnsafePointer[UInt8]()
        self.device = Device()

    @always_inline
    fn __init__(inout self, owned other: Self._pointer_type, owned device: Device):
        self.address = other
        self.device = device

    @no_inline
    fn __str__(self) -> String:
        """Format this pointer as a hexadecimal string.

        Returns:
            A String containing the hexadecimal representation of the memory location
            destination of this pointer.
        """
        return str(self.address)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this pointer address to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write(str(self))

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the DataPointer is *null*.

        Returns:
            Returns False if the DataPointer is *null* and True otherwise.
        """
        return bool(self.address)

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return int(self.address)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.address == rhs.address

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.address != rhs.address

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.address < rhs.address

    @always_inline("nodebug")
    fn offset[T: IntLike](self, idx: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed DataPointer.
        """
        return Self(self.address.offset(idx), self.device)

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


    @always_inline
    fn is_aligned[alignment: Int](self) -> Bool:
        """Checks if the pointer is aligned.

        Parameters:
            alignment: The minimal desired alignment.

        Returns:
            `True` if the pointer is at least `alignment`-aligned or `False`
            otherwise.
        """
        constrained[
            is_power_of_two(alignment), "alignment must be a power of 2."
        ]()
        return int(self) % alignment == 0

    @always_inline("nodebug")
    fn load[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](self) -> SIMD[type, width]:
        return self.address.bitcast[type]().load[width=width, alignment=alignment]()

    @always_inline("nodebug")
    fn store[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](self, owned val: SIMD[type, width]):
        self.address.bitcast[type]().store[width=width, alignment=alignment](val)

    fn load[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](self: Self, offset: Int) -> SIMD[type, width]:
        return (self + int(offset)).load[type=type, width=width, alignment=alignment]()

    fn store[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](self: Self, offset: Int, owned value: SIMD[type, width]):
        (self + int(offset)).store[type=type, width=width, alignment=alignment](value)   

    @always_inline
    fn alloc[alignment: Int = 1,](owned self, count: Int) -> Self:
        var ptr : UnsafePointer[UInt8] = UnsafePointer[UInt8]()
        if self.device.is_cpu():
           ptr = ptr.alloc(count)
        return DataPointer(ptr, self.device)

    fn free(owned self):
        _free(self.address)