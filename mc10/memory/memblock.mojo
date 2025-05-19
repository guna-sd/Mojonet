from memory import UnsafePointer
from sys import sizeof
from mc10.utils.debuggable import abort, asserts, debug
from mc10.utils.time import TimeStamp

from memory import memset, memcpy


@value
struct _BlockMeta:
    """Metadata for memory blocks with detailed tracking information."""

    var id: Int  # Unique identifier for the block
    var flags: UInt8  # Flags for tracking state
    var size: Int  # Size of the memory block in bytes
    var timestamp: TimeStamp  # When the block was allocated/modified
    var owner: StaticString  # Who owns this memory block
    var tag: StaticString  # Optional tag for categorizing memory usage

    # Flag definitions - using bit flags for compact storage
    alias FLAG_ALLOCATED = 0x01  # Memory is currently allocated
    alias FLAG_PINNED = 0x02  # Memory cannot be moved/reallocated
    alias FLAG_EXTERNAL = 0x04  # Memory was externally allocated
    alias FLAG_ZERO_INIT = 0x08  # Memory was zero-initialized
    alias FLAG_LOCKED = 0x40  # Memory is locked and cannot be freed
    alias FLAG_UNINITIALIZED = 0x80  # Memory is uninitialized
    alias FLAG_SHARED = 0x20  # Memory is shared with other components

    @always_inline
    fn __init__(
        out self,
        id: Int = 0,
        flags: UInt8 = 0,
        size: Int = 0,
        owner: StaticString = StaticString("_internal_mem_init"),
        tag: StaticString = StaticString(""),
    ):
        self.id = id
        self.flags = flags
        self.size = size
        self.timestamp = TimeStamp()
        self.owner = owner
        self.tag = tag

    # Helper methods for flag manipulation
    @always_inline
    fn has_flag(self, flag: UInt8) -> Bool:
        return (self.flags & flag) != 0

    @always_inline
    fn set_flag(mut self, flag: UInt8):
        self.flags |= flag

    @always_inline
    fn clear_flag(mut self, flag: UInt8):
        self.flags &= ~flag

    @always_inline
    fn update_timestamp(mut self):
        self.timestamp = TimeStamp()

## TODO: This is still under development and not fully functional as of now.

@value
struct MemBlock[
    *,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    # Instance variables
    var ptr: UnsafePointer[Byte, mut=mut, origin=origin]
    var meta: _BlockMeta

    # ===-------------------------------------------------------------------===#
    # Lifecycle Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Initialize an empty memory block."""
        self.ptr = __type_of(self.ptr)()
        self.meta = _BlockMeta()

    @always_inline
    fn __init__(
        out self,
        size: Int,
        owner: StaticString = StaticString("default"),
        tag: StaticString = StaticString(""),
        zero_init: Bool = False,
    ):
        var alloc_size = max(size, 8)

        var raw_ptr = __type_of(self.ptr).alloc(size)
        if not raw_ptr:
            abort(
                "Memory allocation failed - requested "
                + String(alloc_size)
                + " bytes"
            )

        self.ptr = raw_ptr.origin_cast[mut=mut, origin=origin]()

        var flags = _BlockMeta.FLAG_ALLOCATED
        if zero_init:
            memset(self.ptr, 0, size)
            flags |= _BlockMeta.FLAG_ZERO_INIT

        self.meta = _BlockMeta(
            0,
            flags,
            alloc_size,
            owner,
            tag,
        )

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Byte],
        size: Int,
        is_owned: Bool = False,
        owner: StaticString = StaticString("external"),
        tag: StaticString = StaticString(""),
    ):
        """Create a memory block from an existing pointer.

        Args:
            ptr: Existing pointer.
            size: Size of the memory block.
            is_owned: Whether this block owns the memory (responsible for freeing).
            owner: Identifier for the owner.
            tag: Optional tag for memory tracking.
        """
        debug_assert(ptr, "Cannot create MemBlock from null pointer")
        debug_assert(size > 0, "Memory block size must be positive")

        self.ptr = ptr

        var flags = _BlockMeta.FLAG_ALLOCATED
        if not is_owned:
            flags |= _BlockMeta.FLAG_EXTERNAL

        self.meta = _BlockMeta(0, flags, size, owner, tag)

    @always_inline
    fn deep_copy(self) -> Self:
        """Create a complete copy of this memory block including its contents.

        Returns:
            A new memory block with copied contents.
        """
        if self.meta.size == 0:
            return Self()

        var new_block = __type_of(self)()

        var new_ptr = __type_of(self.ptr).alloc(self.size())

        memcpy(new_ptr, self.ptr, self.size())

        new_block.ptr = new_ptr
        new_block.meta = self.meta

        return new_block

    @always_inline
    fn move(owned self) -> Self:
        """Move this memory block, transferring ownership.

        Returns:
            The moved memory block.
        """
        var new_block = self

        self.ptr = __type_of(self.ptr)()
        self.meta = _BlockMeta()

        return new_block^

    @always_inline
    fn __moveinit__(out self, owned other: Self):
        """Move initialization from another block.

        Args:
            other: The source memory block.
        """
        self = other.move()

    @always_inline
    fn free(mut self):
        """Free the memory block if it's allocated and owned."""
        if self.is_allocated() and not self.is_external():
            if self.is_locked():
                abort("Attempt to free locked memory block")

            # Only free if not null
            if self.ptr:
                self.ptr.free()

            # Reset state
            self.ptr = __type_of(self.ptr)()
            self.meta = _BlockMeta()
            return

    @always_inline
    fn __del__(owned self):
        """Destructor to automatically free memory."""
        self.free()

    # ===-------------------------------------------------------------------===#
    # Memory Operations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn resize(mut self, new_size: Int) -> Bool:
        """Resize the memory block.

        Args:
            new_size: The new size in bytes.
        Returns:
            True if resizing was successful
        """
        # Check conditions that prevent resizing
        if (
            not self.is_allocated()
            or self.is_external()
            or self.is_pinned()
            or self.is_locked()
        ):
            return False

        # No change needed
        if new_size == self.meta.size:
            return True

        var new_block = __type_of(self)()
        var new_ptr = __type_of(self.ptr).alloc(new_size)

        if self.meta.size > 0:
            var copy_size = min(self.meta.size, new_size)
            memcpy(new_block.ptr, self.ptr, copy_size)

        new_block.meta = self.meta
        new_block.meta.update_timestamp()

        # Free old block and replace with new one
        self.free()
        self = new_block
        return True

    @always_inline
    fn fill(mut self, value: Byte = 0):
        """Fill the memory block with a specific value.

        Args:
            value: The byte value to fill with (default 0).
        """
        if self.is_allocated() and self.meta.size > 0:
            memset(self.ptr, value, self.meta.size)

            # Update metadata
            if value == 0:
                self.meta.set_flag(_BlockMeta.FLAG_ZERO_INIT)
            else:
                self.meta.clear_flag(_BlockMeta.FLAG_ZERO_INIT)

            self.meta.update_timestamp()

    @always_inline
    fn zero(mut self):
        """Zero-initialize the memory block."""
        self.fill(0)

    @always_inline
    fn copy_from(mut self, src: Self, count: Int = -1):
        """Copy data from another memory block.

        Args:
            src: Source memory block.
            count: Number of bytes to copy (-1 means use smaller of the two sizes).
        """
        debug_assert(self.is_allocated(), "Destination block not allocated")
        debug_assert(src.is_allocated(), "Source block not allocated")

        var copy_size = count
        if copy_size < 0:
            copy_size = min(self.meta.size, src.meta.size)
        else:
            debug_assert(
                copy_size <= self.meta.size,
                "Copy size exceeds destination capacity",
            )
            debug_assert(
                copy_size <= src.meta.size, "Copy size exceeds source capacity"
            )

        # Perform copy and update metadata
        if copy_size > 0:
            memcpy(self.ptr, src.ptr, copy_size)

            self.meta.update_timestamp()

            # Update zero-init state
            if src.is_zero_initialized() and copy_size == self.meta.size:
                self.meta.set_flag(_BlockMeta.FLAG_ZERO_INIT)
            else:
                self.meta.clear_flag(_BlockMeta.FLAG_ZERO_INIT)

    # ===-------------------------------------------------------------------===#
    # State Query Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn is_allocated(self) -> Bool:
        """Check if the block is allocated."""
        return self.meta.has_flag(_BlockMeta.FLAG_ALLOCATED)

    @always_inline
    fn is_pinned(self) -> Bool:
        """Check if the block is pinned (cannot be moved)."""
        return self.meta.has_flag(_BlockMeta.FLAG_PINNED)

    @always_inline
    fn is_external(self) -> Bool:
        """Check if the block uses externally allocated memory."""
        return self.meta.has_flag(_BlockMeta.FLAG_EXTERNAL)

    @always_inline
    fn is_zero_initialized(self) -> Bool:
        """Check if the block is zero-initialized."""
        return self.meta.has_flag(_BlockMeta.FLAG_ZERO_INIT)

    @always_inline
    fn is_shared(self) -> Bool:
        """Check if the block is shared across contexts."""
        return self.meta.has_flag(_BlockMeta.FLAG_SHARED)

    @always_inline
    fn is_locked(self) -> Bool:
        """Check if the block is locked from modification."""
        return self.meta.has_flag(_BlockMeta.FLAG_LOCKED)

    @always_inline
    fn is_uninitialized(self) -> Bool:
        """Check if the block memory is uninitialized."""
        return self.meta.has_flag(_BlockMeta.FLAG_UNINITIALIZED)

    @always_inline
    fn size(self) -> Int:
        """Get the size of the memory block in bytes."""
        return self.meta.size

    @always_inline
    fn owner(self) -> StaticString:
        """Get the owner identifier of the memory block."""
        return self.meta.owner

    @always_inline
    fn tag(self) -> StaticString:
        """Get the tag of the memory block."""
        return self.meta.tag

    @always_inline
    fn timestamp(self) -> TimeStamp:
        """Get the timestamp of the memory block's last modification."""
        return self.meta.timestamp

    # ===-------------------------------------------------------------------===#
    # State Modification Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn pin(mut self):
        """Pin the memory block so it cannot be moved."""
        self.meta.set_flag(_BlockMeta.FLAG_PINNED)

    @always_inline
    fn unpin(mut self):
        """Unpin the memory block."""
        self.meta.clear_flag(_BlockMeta.FLAG_PINNED)

    @always_inline
    fn lock(mut self):
        """Lock the memory block to prevent modifications."""
        self.meta.set_flag(_BlockMeta.FLAG_LOCKED)

    @always_inline
    fn unlock(mut self):
        """Unlock the memory block."""
        self.meta.clear_flag(_BlockMeta.FLAG_LOCKED)

    @always_inline
    fn mark_shared(mut self):
        """Mark this memory as shared across contexts."""
        self.meta.set_flag(_BlockMeta.FLAG_SHARED)

    @always_inline
    fn set_owner(mut self, owner: StaticString):
        """Set the owner identifier for this memory block."""
        self.meta.owner = owner
        self.meta.update_timestamp()

    @always_inline
    fn set_tag(mut self, tag: StaticString):
        """Set the tag for this memory block."""
        self.meta.tag = tag

    # ===-------------------------------------------------------------------===#
    # Pointer Access Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn offset(self, offset: Int) -> __type_of(self.ptr):
        """Get a pointer offset from the base address.

        Args:
            offset: Byte offset from the start of the block.
        Returns:
            Pointer at the requested offset.
        """
        debug_assert(
            self.is_allocated(), "Cannot get pointer from unallocated block"
        )
        debug_assert(
            offset >= 0 and offset < self.meta.size, "Offset out of bounds"
        )
        return self.ptr + offset

    @always_inline
    fn get_typed_ptr[
        T: AnyType
    ](self) -> UnsafePointer[
        T,
        mut=mut,
        origin=origin,
    ]:
        """Get a typed pointer to the memory block.

        Returns:
            A pointer with the requested type.
        """
        debug_assert(
            self.is_allocated(), "Cannot get pointer from unallocated block"
        )

        return self.ptr.bitcast[T]()

    @always_inline
    fn get_typed_ptr[
        T: AnyType
    ](self, offset: Int) -> UnsafePointer[
        T,
        mut=mut,
        origin=origin,
    ]:
        """Get a typed pointer at an offset from the base address.

        Args:
            offset: Element offset (not byte offset).
        Returns:
            A typed pointer at the requested offset.
        """
        debug_assert(
            self.is_allocated(), "Cannot get pointer from unallocated block"
        )

        var byte_offset = offset * sizeof[T]()
        debug_assert(
            byte_offset >= 0 and byte_offset < self.meta.size,
            "Element offset out of bounds",
        )

        var ptr = self.ptr + byte_offset

        return ptr.bitcast[T]()

    # ===-------------------------------------------------------------------===#
    # Static Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn merge(mut first: Self, mut second: Self) -> Bool:
        """Merge two consecutive memory blocks if possible.

        Args:
            first: The first memory block.
            second: The second memory block.
        Returns:
            True if blocks were merged, False otherwise.
        """
        # Check conditions that allow merging
        if not first.is_allocated() or not second.is_allocated():
            return False

        # Verify blocks are actually adjacent in memory
        if first.ptr + first.meta.size != second.ptr:
            return False

        # Other conditions that prevent merging
        if (
            first.is_external() != second.is_external()
            or first.is_pinned()
            or second.is_pinned()
            or first.is_locked()
            or second.is_locked()
            or first.is_shared()
            or second.is_shared()
            or first.meta.owner != second.meta.owner
        ):
            return False

        # Merge sizes
        first.meta.size += second.meta.size
        first.meta.update_timestamp()

        # Update zero-init status
        if not first.is_zero_initialized() or not second.is_zero_initialized():
            first.meta.clear_flag(_BlockMeta.FLAG_ZERO_INIT)

        # Clear second block's allocation flag without freeing memory
        if not second.is_external():
            second.meta.clear_flag(_BlockMeta.FLAG_ALLOCATED)
            second.meta.size = 0

        return True

    @staticmethod
    fn create_zero_initialized(
        size: Int,
        owner: StaticString = StaticString("default"),
        tag: StaticString = StaticString("zero_init"),
    ) -> Self:
        """Create a zero-initialized memory block.

        Args:
            size: Size in bytes to allocate.
            owner: Identifier for the owner.
            tag: Optional tag.
        Returns:
            A new zero-initialized memory block
        """
        var block = Self(size, owner, tag, True)
        return block

    @staticmethod
    fn from_external_pointer(
        ptr: UnsafePointer[Byte],
        size: Int,
        owner: StaticString = StaticString("external"),
        tag: StaticString = StaticString(""),
    ) -> Self:
        """Create a memory block from an external pointer.

        Args:
            ptr: External pointer.
            size: Size of the memory region.
            owner: Identifier for the owner.
            tag: Optional tag.
        Returns:
            A memory block wrapper for the external pointer.
        """
        return Self(ptr, size, False, owner, tag)
