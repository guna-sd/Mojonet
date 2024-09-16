from memory.unsafe_pointer import alignof, _free, is_power_of_two
from memory.memory import memcpy, _malloc
from gpu.host.memory import _malloc as gpu_malloc


struct Allocator:

    @staticmethod
    @always_inline
    fn allocate[alignment: Int = 1](size: Int, device: Device = Device.CPU) -> DataPointer:
        """Allocates a block of memory with a specified size and alignment.

        Parameters:
            alignment: The alignment requirement of the memory block.

        Args:
            size: The size of the memory block to allocate in bytes.
            device: The device to which the memory block is to be allocated.

        Returns:
            A DataPointer pointing to the allocated memory.
        """
        constrained[
            is_power_of_two(alignment), "alignment must be a power of 2."
        ]()
        var ptr : UnsafePointer[UInt8] = UnsafePointer[UInt8]()
        if device.is_cpu():
           ptr = _malloc[UInt8, alignment=alignment](size)
        return DataPointer(ptr, device)

    @staticmethod
    @always_inline
    fn deallocate(data_pointer: DataPointer):
        """Deallocates a block of memory.

        Args:
            data_pointer: The DataPointer pointing to the memory to deallocate.
        """
        if data_pointer.device.is_cpu():
            _free(data_pointer.address)

    @staticmethod
    @always_inline
    fn reallocate[alignment: Int = 1](inout data_pointer: DataPointer, new_size: Int, old_size: Int):
        """Reallocates a block of memory to a new size with a specified alignment.

        Parameters:
            alignment: The alignment requirement of the memory block.

        Args:
            data_pointer: The DataPointer pointing to the memory to reallocate.
            new_size: The new size of the memory block in bytes.
            old_size: The size of the memory allocated previosly.
        """
        constrained[
            is_power_of_two(alignment), "alignment must be a power of 2."
        ]()
        var new_ptr = UnsafePointer[UInt8]()
        new_ptr =  _malloc[UInt8, alignment=alignment](new_size)
        memcpy(new_ptr, data_pointer.address, old_size)
        data_pointer = DataPointer(new_ptr, data_pointer.device)