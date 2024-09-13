from memory.unsafe_pointer import is_power_of_two


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
    fn __init__(inout self, other: Self._pointer_type, device: Device):
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