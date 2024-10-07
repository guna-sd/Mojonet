@value
@register_passable("trivial")
struct DataPointer:
    alias __ptr_type = UnsafePointer[UInt8]
    alias Null = Self.__ptr_type()
    var ptr: Self.__ptr_type
    var dtype: DType

    fn __init__(inout self):
        self.dtype = DType.invalid
        self.ptr = Self.Null
    
    fn __init__(inout self, dtype: DType):
        self.dtype = dtype
        self.ptr = Self.Null
    
    fn __init__(inout self, owned ptr: Self.__ptr_type, owned dtype: DType):
        self.ptr = ptr
        self.dtype = dtype

    @always_inline
    fn alloc(inout self, count: Int):
        var sizeof_t : Int = __sizeof(self.dtype)
        if sizeof_t == -1 or sizeof_t == 0:
            handle_issue("DType invalid could not allocate memory")
        if self.dtype == DType.uint8:
            self.ptr = self.ptr.alloc(count)
            return
        self.ptr = self.ptr.alloc(sizeof_t * count)
        return

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

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the DataPointer is *null*.

        Returns:
            Returns False if the DataPointer is *null* and True otherwise.
        """
        return bool(self.ptr)

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return int(self.ptr)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.ptr == rhs.ptr

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.ptr != rhs.ptr

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.ptr < rhs.ptr

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
        return Self(self.ptr.offset(idx), self.dtype)

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

    fn __getitem__[
        Type: CollectionElement,
    ](self, offset: Int) -> Type:
        return (self.ptr.bitcast[Type]() + offset)[]

    fn __setitem__[
        Type: CollectionElement
    ](self, offset: Int, owned value: Type):
        (self.ptr.bitcast[Type]() + offset).init_pointee_move(value)