from memory import UnsafePointer
from mc10.hardware.device import Device
from mc10.core.pointer import DataPointer, default_delete
from mc10.utils.Function import Function
from .utils import ALIGNMENT
from mc10.core.context import GlobalContext
from os import abort


fn register(allocator: Some[Allocator]):
    GlobalContext.register(allocator)


@register_passable("trivial")
trait Allocator(ImplicitlyCopyable, Movable):
    """Abstract allocator trait for memory allocation.

    Similar to PyTorch's Allocator interface, this trait defines the interface
    for memory allocation and deallocation on different devices.
    """

    @staticmethod
    fn allocate_raw(size: Int) -> OpaquePointer[MutOrigin.external]:
        """Allocate memory on the device.

        Args:
            size: Number of bytes to allocate.

        Returns:
            An OpaquePointer to the allocated memory.
        """
        ...

    @staticmethod
    fn free_raw(mut ptr: OpaquePointer[MutOrigin.external]) -> None:
        """Deallocate memory."""
        ...

    @staticmethod
    fn allocate(size: Int) -> DataPointer:
        var ptr = Self.allocate_raw(size)
        if ptr == OpaquePointer[MutOrigin.external]():
            return abort[DataPointer](
                "Memory allocation failed for " + String(size) + " bytes"
            )
        return DataPointer(
            data=ptr, ctx=ptr, deleter=Self.free, device=Self.device()
        )

    @staticmethod
    fn free(mut ptr: OpaquePointer[MutOrigin.external]) -> None:
        """Deallocate memory.

        Usually do nothing here - DataPointer.__del__ calls stored deleter.
        Caching allocators override to intercept and cache.
        """
        Self.free_raw(ptr)

    @staticmethod
    fn device() -> Device:
        """Get the device this allocator manages.

        Returns:
            The Device this allocator operates on.
        """
        ...

    @staticmethod
    fn name() -> StaticString:
        return "UnknownAllocator"

    @staticmethod
    fn getAllocatorImpl() -> AllocatorImpl:
        return AllocatorImpl(
            _raw_alloc=Self.allocate_raw,
            _raw_free=Self.free_raw,
            _smart_alloc=Self.allocate,
            _smart_free=Self.free,
            _device=Self.device(),
            _name=Self.name(),
        )


# Still got a bunch of refinement over here...
# This is a temporary work around later add the vtable , object , interface implementaion...
# Refer https://github.com/YichengDWu/interface ... this is a great work around to look for....


@fieldwise_init
@register_passable("trivial")
struct AllocatorImpl(
    Copyable, ImplicitlyCopyable, Movable, Representable, Stringable
):
    """
    The Pluggable Allocator Interface Implementation.
    Wraps an Allocator trait object and provides a uniform interface for allocation.
    """

    comptime _SmartAllocFn = fn (size: Int) -> DataPointer
    comptime _RawAllocFn = fn (size: Int) -> OpaquePointer[MutOrigin.external]
    comptime _RawFreeFn = fn (
        mut ptr: OpaquePointer[MutOrigin.external]
    ) -> None
    comptime _SmartFreeFn = fn (
        mut ptr: OpaquePointer[MutOrigin.external]
    ) -> None

    var _raw_alloc: Function[Self._RawAllocFn]
    var _raw_free: Function[Self._RawFreeFn]
    var _smart_alloc: Function[Self._SmartAllocFn]
    var _smart_free: Function[Self._SmartFreeFn]
    var _device: Device
    var _name: StaticString

    @implicit
    fn __init__(out self, allocator: Some[Allocator]):
        self = allocator.getAllocatorImpl()

    @always_inline
    fn __str__(self) -> String:
        return self._name

    @always_inline
    fn __repr__(self) -> String:
        return String(self)

    @staticmethod
    fn get_default() -> Self:
        return DefaultAllocator.getAllocatorImpl()

    @staticmethod
    fn fromAllocator(allocator: Some[Allocator]) -> AllocatorImpl:
        return allocator.getAllocatorImpl()


comptime DefaultAllocator = CPUAllocator


@register_passable("trivial")
struct CPUAllocator(Allocator):
    """CPU memory allocator using libc malloc/free.

    Manages heap memory allocation on CPU. Uses standard libc malloc/free
    internally with proper device tracking.
    """

    fn __init__(out self):
        """Initialize CPU allocator."""
        pass

    @staticmethod
    fn allocate_raw(size: Int) -> OpaquePointer[MutOrigin.external]:
        return alloc[NoneType](size)

    @staticmethod
    fn allocate(
        size: Int,
    ) -> DataPointer:
        """Allocate memory on CPU.

        Args:
            size: Number of bytes to allocate.

        """
        var ptr = Self.allocate_raw(size)
        if ptr == OpaquePointer[MutOrigin.external]():
            return abort[DataPointer](
                "Memory allocation failed for " + String(size) + " bytes"
            )
        return DataPointer(
            data=ptr, ctx=ptr, deleter=Self.free, device=Self.device()
        )

    @staticmethod
    fn device() -> Device:
        """Get the device (CPU).

        Returns:
            Device.CPU.
        """
        return Device.CPU

    @staticmethod
    fn free_raw(mut ptr: OpaquePointer[MutOrigin.external]) -> None:
        """Deallocate memory."""
        ptr.free()

    @staticmethod
    fn name() -> String:
        return "CPUAllocator"


# struct CUDAAllocater(Allocator):
#     """
#     Placeholder for CUDA GPU memory.
#     """

#     alias cuda_malloc_fn = fn (
#         ptr_out: UnsafePointer[UnsafePointer[UInt8]], size: Int
#     ) -> Int32
#     alias cuda_free_fn = fn (ptr: UnsafePointer[UInt8]) -> Int32
#     alias cudaSuccess = 0

#     var device: Device

#     fn __init__(out self, device: Device):
#         self.device = device

#     fn alloc(self, size: Int, align: Int = ALIGNMENT) -> UnsafePointer[Byte]:
#         # TODO: Insert ExternalCall to cudaMalloc here
#         print("CUDA alloc not implemented")
#         return UnsafePointer[Byte]()  # Null

#     fn free(self, ptr: UnsafePointer[Byte]):
#         # TODO: Insert ExternalCall to cudaFree here
#         print("CUDA free not implemented")

#     fn device(self) -> Device:
#         return Device.CUDA


# struct AMDAllocator(Allocator):
#     """
#     Placeholder for AMD GPU memory.
#     """

#     var device_id: Int

#     fn __init__(out self, device_id: Int):
#         self.device_id = device_id

#     fn alloc(self, size: Int, align: Int = ALIGNMENT) -> UnsafePointer[Byte]:
#         # TODO: Insert ExternalCall...
#         print("AMD alloc not implemented")
#         return UnsafePointer[Byte]()  # Null

#     fn free(self, ptr: UnsafePointer[Byte]):
#         # TODO: Insert ExternalCall to cudaFree here
#         print("AMD free not implemented")

#     fn device(self) -> Device:
#         return Device.AMD
