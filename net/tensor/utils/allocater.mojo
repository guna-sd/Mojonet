from .device import Device

@value
struct DataPtr:
    """Represents a pointer with an associated device."""
    
    var data: UnsafePointer[UInt8]
    var device: Device

    fn __init__(inout self, data: UnsafePointer[UInt8], device: Device = Device.CPU):
        """Initializes a DataPtr with a data pointer and a device."""
        self.data = data
        self.device = device

    fn clear(owned self):
        """Frees the associated memory and clears the data pointer."""
        if self.data:
            self.data.free()

    fn get(self) -> UnsafePointer[UInt8]:
        """Returns the data pointer."""
        return self.data

    fn is_valid(self) -> Bool:
        """Checks if the data pointer is valid (i.e., not null)."""
        return self.data


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
        var data: UnsafePointer[UInt8] = UnsafePointer[UInt8].alloc(size)
        return DataPtr(data, device)

    @staticmethod
    fn free(owned data_ptr: DataPtr):
        """Frees the memory held by the given DataPtr.

        Args:
            data_ptr: The DataPtr whose memory is to be freed.
        """
        data_ptr.clear()