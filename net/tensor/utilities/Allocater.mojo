struct Allocator:
    """Represents an allocator that allocates and frees memory."""

    @staticmethod
    fn allocate(size: Int, device: Device = Device.CPU) -> DataPointer:
        """Allocates memory of the given size for a specific device.

        Args:
            size: The size in bytes of the memory to allocate.
            device: The device on which to allocate memory (defaults to CPU).

        Returns:
            A DataPtr that holds the allocated memory and device.
        """
        var ptr: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(size)
        return DataPointer(ptr, device)

    @staticmethod
    fn free(owned ptr: DataPointer):
        """Frees the memory held by the given DataPtr.

        Args:
            ptr: The DataPtr whose memory is to be freed.
        """
        ptr.free()
