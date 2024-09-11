@register_passable("trivial")
struct DataPtr:
    """Represents a pointer with an associated device."""
    
    var ptr: UnsafePointer[UInt8]
    var device: Device


    @always_inline
    fn __init__(inout self):
        """Create a null pointer."""
        self.ptr = UnsafePointer[UInt8]()
        self.device = Device()

    @always_inline
    fn __init__(inout self, ptr: UnsafePointer[UInt8], device: Device = Device.CPU):
        """Initializes a DataPtr with a data pointer and a device."""
        self.ptr = ptr
        self.device = device

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        self.ptr = other.ptr
        self.device = other.device

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return int(self.ptr) != 0

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the pointer.

        Returns:
            The string representation of the pointer.
        """
        return hex(int(self.ptr))

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this pointer address to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write(str(self))

    fn free(owned self):
        """Frees the associated memory and clears the data pointer."""
        if self.ptr:
            self.ptr.free()

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns the data pointer."""
        return self.ptr

    fn is_valid(self) -> Bool:
        """Checks if the data pointer is valid (i.e., not null)."""
        return self.ptr


struct Allocator:
    """Represents an allocator that allocates and frees memory."""
    
    @staticmethod
    fn allocate(size: Int, device: Device = Device.CPU) -> DataPtr:
        """Allocates memory of the given size for a specific device.

        Args:
            size: The size in bytes of the memory to allocate.
            device: The device on which to allocate memory (defaults to CPU).

        Returns:
            A DataPtr that holds the allocated memory and device.
        """
        var ptr: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(size)
        return DataPtr(ptr, device)

    @staticmethod
    fn free(owned ptr: DataPtr):
        """Frees the memory held by the given DataPtr.

        Args:
            ptr: The DataPtr whose memory is to be freed.
        """
        ptr.free()


@value
@register_passable("trivial")
struct DataPointer(Boolable, CollectionElement, Intable, Stringable, EqualityComparable):
    alias _pointer_type = UnsafePointer[UInt8]
    var address: Self._pointer_type
    var device: Device
    """The pointed-to address."""

    @always_inline
    fn __init__(inout self):
        self.address = Self._pointer_type()
        self.device = Device()

    @always_inline
    fn __init__(inout self, other: UnsafePointer[UInt8], device: Device):
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
        return self.address.__bool__()

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return int(self.address)

    @always_inline
    fn __getitem__[type: DType](self, offset: Int) -> Scalar[type]:
        """Loads a single element (SIMD of size 1) from the pointer at the
        specified index.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.address.load(offset).cast[type]()

    @always_inline
    fn __setitem__[type: DType](self, offset: Int, val: Scalar[type]):
        """Stores a single element value at the given offset.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        return self.address.store(offset, val.cast[DType.uint8]())

    # ===------------------------------------------------------------------=== #
    # Comparisons
    # ===------------------------------------------------------------------=== #

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

    # ===------------------------------------------------------------------=== #
    # Allocate/Free
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn alloc[type: DType, alignment: Int = alignof[type]()](count: Int, /, *,device: Device) -> Self:
        """Heap-allocates a number of element of the specified type using
        the specified alignment.

        Parameter:
            alignment: The alignment used for the allocation.

        Args:
            count: The number of elements to allocate (note that this is not
              the bytecount).
            device: The device to be allocated.

        Returns:
            A new `DataPointer` object which has been allocated on the heap.
        """
        return Self(_malloc[Scalar[type], alignment=alignment](
            count * sizeof[type]()
        ).bitcast[UInt8](), device)

    @always_inline
    fn free(self):
        """Frees the heap allocates memory."""
        _free(self.address)

    # ===------------------------------------------------------------------=== #
    # Casting
    # ===------------------------------------------------------------------=== #

    # @always_inline("nodebug")
    # fn bitcast[
    #     new_type: DType = type,
    #     /,
    #     address_space: AddressSpace = Self.address_space,
    # ](self) -> DTypePointer[new_type, address_space]:
    #     """Bitcasts `DTypePointer` to a different dtype.

    #     Parameters:
    #         new_type: The target dtype.
    #         address_space: The address space of the result.

    #     Returns:
    #         A new `DTypePointer` object with the specified dtype and the same
    #         address, as the original `DTypePointer`.
    #     """
    #     return self.address.bitcast[SIMD[new_type, 1], address_space]()

    # @always_inline("nodebug")
    # fn _as_scalar_pointer(self) -> Pointer[Scalar[type], address_space]:
    #     """Converts the `DTypePointer` to a scalar pointer of the same dtype.

    #     Returns:
    #         A `Pointer` to a scalar of the same dtype.
    #     """
    #     return self.address

    # # ===------------------------------------------------------------------=== #
    # # Load/Store
    # # ===------------------------------------------------------------------=== #

    # alias _default_alignment = alignof[
    #     Scalar[type]
    # ]() if triple_is_nvidia_cuda() else 1

    # @always_inline
    # fn prefetch[params: PrefetchOptions](self):
    #     # Prefetch at the underlying address.
    #     """Prefetches memory at the underlying address.

    #     Parameters:
    #         params: Prefetch options (see `PrefetchOptions` for details).
    #     """
    #     _prefetch[params](self)

    # @always_inline("nodebug")
    # fn load[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](self) -> SIMD[type, width]:
    #     """Loads the value the Pointer object points to.

    #     Constraints:
    #         The width and alignment must be positive integer values.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Returns:
    #         The loaded value.
    #     """
    #     return self.load[width=width, alignment=alignment](0)

    # @always_inline("nodebug")
    # fn load[
    #     T: Intable, *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](self, offset: T) -> SIMD[type, width]:
    #     """Loads the value the Pointer object points to with the given offset.

    #     Constraints:
    #         The width and alignment must be positive integer values.

    #     Parameters:
    #         T: The Intable type of the offset.
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The offset to load from.

    #     Returns:
    #         The loaded value.
    #     """

    #     @parameter
    #     if triple_is_nvidia_cuda() and sizeof[type]() == 1 and alignment == 1:
    #         # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
    #         # regardless of the alignment that is passed. This causes issues if
    #         # this method is called on an unaligned pointer.
    #         # TODO #37823 We can make this smarter when we add an `aligned`
    #         # trait to the pointer class.
    #         var v = SIMD[type, width]()

    #         # intentionally don't unroll, otherwise the compiler vectorizes
    #         for i in range(width):
    #             v[i] = self.address.offset(int(offset) + i).load[
    #                 alignment=alignment
    #             ]()
    #         return v

    #     return (
    #         self.address.offset(offset)
    #         .bitcast[SIMD[type, width]]()
    #         .load[alignment=alignment]()
    #     )

    # @always_inline("nodebug")
    # fn store[
    #     T: Intable,
    #     /,
    #     *,
    #     width: Int = 1,
    #     alignment: Int = Self._default_alignment,
    # ](self, offset: T, val: SIMD[type, width]):
    #     """Stores a single element value at the given offset.

    #     Constraints:
    #         The width and alignment must be positive integer values.

    #     Parameters:
    #         T: The Intable type of the offset.
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The offset to store to.
    #         val: The value to store.
    #     """
    #     self.offset(offset).store[width=width, alignment=alignment](val)

    # @always_inline("nodebug")
    # fn store[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](self, val: SIMD[type, width]):
    #     """Stores a single element value.

    #     Constraints:
    #         The width and alignment must be positive integer values.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         val: The value to store.
    #     """
    #     constrained[width > 0, "width must be a positive integer value"]()
    #     constrained[
    #         alignment > 0, "alignment must be a positive integer value"
    #     ]()
    #     self.address.bitcast[SIMD[type, width]]().store[alignment=alignment](
    #         val
    #     )

    # @always_inline("nodebug")
    # fn simd_nt_store[
    #     width: Int, T: Intable
    # ](self, offset: T, val: SIMD[type, width]):
    #     """Stores a SIMD vector using non-temporal store.

    #     Parameters:
    #         width: The SIMD width.
    #         T: The Intable type of the offset.

    #     Args:
    #         offset: The offset to store to.
    #         val: The SIMD value to store.
    #     """
    #     self.offset(offset).simd_nt_store[width](val)

    # @always_inline("nodebug")
    # fn simd_strided_load[
    #     width: Int, T: Intable
    # ](self, stride: T) -> SIMD[type, width]:
    #     """Performs a strided load of the SIMD vector.

    #     Parameters:
    #         width: The SIMD width.
    #         T: The Intable type of the stride.

    #     Args:
    #         stride: The stride between loads.

    #     Returns:
    #         A vector which is stride loaded.
    #     """
    #     return strided_load[type, width](
    #         self, int(stride), SIMD[DType.bool, width](1)
    #     )

    # @always_inline("nodebug")
    # fn simd_strided_store[
    #     width: Int, T: Intable
    # ](self, val: SIMD[type, width], stride: T):
    #     """Performs a strided store of the SIMD vector.

    #     Parameters:
    #         width: The SIMD width.
    #         T: The Intable type of the stride.

    #     Args:
    #         val: The SIMD value to store.
    #         stride: The stride between stores.
    #     """
    #     strided_store(val, self, int(stride), True)

    # @always_inline("nodebug")
    # fn simd_nt_store[width: Int](self, val: SIMD[type, width]):
    #     """Stores a SIMD vector using non-temporal store.

    #     The address must be properly aligned, 64B for avx512, 32B for avx2, and
    #     16B for avx.

    #     Parameters:
    #         width: The SIMD width.

    #     Args:
    #         val: The SIMD value to store.
    #     """
    #     # Store a simd value into the pointer. The address must be properly
    #     # aligned, 64B for avx512, 32B for avx2, and 16B for avx.
    #     self.address.bitcast[SIMD[type, width]]().nt_store(val)

    # # ===------------------------------------------------------------------=== #
    # # Gather/Scatter
    # # ===------------------------------------------------------------------=== #

    # @always_inline("nodebug")
    # fn gather[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](self, offset: SIMD[_, width]) -> SIMD[type, width]:
    #     """Gathers a SIMD vector from offsets of the current pointer.

    #     This method loads from memory addresses calculated by appropriately
    #     shifting the current pointer according to the `offset` SIMD vector.

    #     Constraints:
    #         The offset type must be an integral type.
    #         The alignment must be a power of two integer value.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The SIMD vector of offsets to gather from.

    #     Returns:
    #         The SIMD vector containing the gathered values.
    #     """
    #     var mask = SIMD[DType.bool, width](True)
    #     var default = SIMD[type, width]()
    #     return self.gather[width=width, alignment=alignment](
    #         offset, mask, default
    #     )

    # @always_inline("nodebug")
    # fn gather[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](
    #     self,
    #     offset: SIMD[_, width],
    #     mask: SIMD[DType.bool, width],
    #     default: SIMD[type, width],
    # ) -> SIMD[type, width]:
    #     """Gathers a SIMD vector from offsets of the current pointer.

    #     This method loads from memory addresses calculated by appropriately
    #     shifting the current pointer according to the `offset` SIMD vector,
    #     or takes from the `default` SIMD vector, depending on the values of
    #     the `mask` SIMD vector.

    #     If a mask element is `True`, the respective result element is given
    #     by the current pointer and the `offset` SIMD vector; otherwise, the
    #     result element is taken from the `default` SIMD vector.

    #     Constraints:
    #         The offset type must be an integral type.
    #         The alignment must be a power of two integer value.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The SIMD vector of offsets to gather from.
    #         mask: The SIMD vector of boolean values, indicating for each
    #             element whether to load from memory or to take from the
    #             `default` SIMD vector.
    #         default: The SIMD vector providing default values to be taken
    #             where the `mask` SIMD vector is `False`.

    #     Returns:
    #         The SIMD vector containing the gathered values.
    #     """
    #     constrained[
    #         offset.type.is_integral(),
    #         "offset type must be an integral type",
    #     ]()
    #     constrained[
    #         is_power_of_two(alignment),
    #         "alignment must be a power of two integer value",
    #     ]()

    #     var base = offset.cast[DType.index]().fma(sizeof[type](), int(self))
    #     return gather(base.cast[DType.address](), mask, default, alignment)

    # @always_inline("nodebug")
    # fn scatter[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](self, offset: SIMD[_, width], val: SIMD[type, width]):
    #     """Scatters a SIMD vector into offsets of the current pointer.

    #     This method stores at memory addresses calculated by appropriately
    #     shifting the current pointer according to the `offset` SIMD vector.

    #     If the same offset is targeted multiple times, the values are stored
    #     in the order they appear in the `val` SIMD vector, from the first to
    #     the last element.

    #     Constraints:
    #         The offset type must be an integral type.
    #         The alignment must be a power of two integer value.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The SIMD vector of offsets to scatter into.
    #         val: The SIMD vector containing the values to be scattered.
    #     """
    #     var mask = SIMD[DType.bool, width](True)
    #     self.scatter[width=width, alignment=alignment](offset, val, mask)

    # @always_inline("nodebug")
    # fn scatter[
    #     *, width: Int = 1, alignment: Int = Self._default_alignment
    # ](
    #     self,
    #     offset: SIMD[_, width],
    #     val: SIMD[type, width],
    #     mask: SIMD[DType.bool, width],
    # ):
    #     """Scatters a SIMD vector into offsets of the current pointer.

    #     This method stores at memory addresses calculated by appropriately
    #     shifting the current pointer according to the `offset` SIMD vector,
    #     depending on the values of the `mask` SIMD vector.

    #     If a mask element is `True`, the respective element in the `val` SIMD
    #     vector is stored at the memory address defined by the current pointer
    #     and the `offset` SIMD vector; otherwise, no action is taken for that
    #     element in `val`.

    #     If the same offset is targeted multiple times, the values are stored
    #     in the order they appear in the `val` SIMD vector, from the first to
    #     the last element.

    #     Constraints:
    #         The offset type must be an integral type.
    #         The alignment must be a power of two integer value.

    #     Parameters:
    #         width: The SIMD width.
    #         alignment: The minimal alignment of the address.

    #     Args:
    #         offset: The SIMD vector of offsets to scatter into.
    #         val: The SIMD vector containing the values to be scattered.
    #         mask: The SIMD vector of boolean values, indicating for each
    #             element whether to store at memory or not.
    #     """
    #     constrained[
    #         offset.type.is_integral(),
    #         "offset type must be an integral type",
    #     ]()
    #     constrained[
    #         is_power_of_two(alignment),
    #         "alignment must be a power of two integer value",
    #     ]()

    #     var base = offset.cast[DType.index]().fma(sizeof[type](), int(self))
    #     scatter(val, base.cast[DType.address](), mask, alignment)

    # @always_inline
    # fn is_aligned[alignment: Int](self) -> Bool:
    #     """Checks if the pointer is aligned.

    #     Parameters:
    #         alignment: The minimal desired alignment.

    #     Returns:
    #         `True` if the pointer is at least `alignment`-aligned or `False`
    #         otherwise.
    #     """
    #     constrained[
    #         is_power_of_two(alignment), "alignment must be a power of 2."
    #     ]()
    #     return int(self) % alignment == 0

    # # ===------------------------------------------------------------------=== #
    # # Pointer Arithmetic
    # # ===------------------------------------------------------------------=== #

    # @always_inline("nodebug")
    # fn offset[T: Intable](self, idx: T) -> Self:
    #     """Returns a new pointer shifted by the specified offset.

    #     Parameters:
    #         T: The Intable type of the offset.

    #     Args:
    #         idx: The offset of the new pointer.

    #     Returns:
    #         The new constructed DTypePointer.
    #     """
    #     return self.address.offset(idx)

    # @always_inline("nodebug")
    # fn __add__[T: Intable](self, rhs: T) -> Self:
    #     """Returns a new pointer shifted by the specified offset.

    #     Parameters:
    #         T: The Intable type of the offset.

    #     Args:
    #         rhs: The offset.

    #     Returns:
    #         The new DTypePointer shifted by the offset.
    #     """
    #     return self.offset(rhs)

    # @always_inline("nodebug")
    # fn __sub__[T: Intable](self, rhs: T) -> Self:
    #     """Returns a new pointer shifted back by the specified offset.

    #     Parameters:
    #         T: The Intable type of the offset.

    #     Args:
    #         rhs: The offset.

    #     Returns:
    #         The new DTypePointer shifted by the offset.
    #     """
    #     return self.offset(-int(rhs))

    # @always_inline("nodebug")
    # fn __iadd__[T: Intable](inout self, rhs: T):
    #     """Shifts the current pointer by the specified offset.

    #     Parameters:
    #         T: The Intable type of the offset.

    #     Args:
    #         rhs: The offset.
    #     """
    #     self = self + rhs

    # @always_inline("nodebug")
    # fn __isub__[T: Intable](inout self, rhs: T):
    #     """Shifts back the current pointer by the specified offset.

    #     Parameters:
    #         T: The Intable type of the offset.

    #     Args:
    #         rhs: The offset.
    #     """
    #     self = self - rhs
