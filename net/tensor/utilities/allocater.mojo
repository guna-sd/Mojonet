from memory.unsafe_pointer import alignof, _free, is_power_of_two
from memory.memory import memcpy, _malloc
from gpu.host.memory import _malloc as gpu_malloc


struct Allocator:

    @staticmethod
    @always_inline
    fn allocate[Type: AnyType](size: Int, device: Device = Device.CPU) -> UnsafePointer[Type]:
        """Allocates a block of memory with a specified size and alignment.

        Parameters:
            Type: The type of the memory to be allocated.

        Args:
            size: The size of the memory block to allocate in bytes.
            device: The device to which the memory block is to be allocated.

        Returns:
            A DataPointer pointing to the allocated memory.
        """
        var ptr : UnsafePointer[Type] = UnsafePointer[Type]()
        if device.is_cpu():
           ptr = ptr.alloc(size)
        return ptr

    @staticmethod
    @always_inline
    fn deallocate(owned storage: StorageImpl):
        """Deallocates a block of memory.

        Args:
            storage: The storage to which the memory block is to be deallocated.
        """
        if storage.device.is_cpu():
            storage.raw_ptr[].free()

    @staticmethod
    @always_inline
    fn reallocate[Type: AnyType, //, *, ](inout data_pointer: UnsafePointer[Type], new_size: Int, old_size: Int):
        """Reallocates a block of memory to a new size with a specified alignment.

        Args:
            data_pointer: The DataPointer pointing to the memory to reallocate.
            new_size: The new size of the memory block in bytes.
            old_size: The size of the memory allocated previosly.
        """
        var new_ptr = UnsafePointer[Type]()
        new_ptr =  new_ptr.alloc(new_size)
        memcpy(new_ptr, data_pointer, old_size)
        data_pointer = new_ptr