from mc10._memory.allocator import (
    Allocator,
    AllocatorImpl,
    DefaultAllocator,
    CPUAllocator,
)
from mc10.hardware.device import Device
from memory import OwnedPointer
from hashlib import Hasher, default_comp_time_hasher, default_hasher
from sys.ffi import _get_global, _Global
from os import abort


struct GlobalContext(Copyable, Movable):
    # ------------------------------------------------------------------ #
    # Global storage handle
    # ------------------------------------------------------------------ #

    # Single global instance backing storage for GlobalContext
    comptime _global_ctx_state = _Global[
        "mojonet.global_context", GlobalContext._init_global_context
    ]

    # ------------------------------------------------------------------ #
    # Allocator registry types
    # ------------------------------------------------------------------ #

    var _allocator: Dict[Device, AllocatorImpl]

    fn __init__(out self, allocator: AllocatorImpl):
        self._allocator = {allocator._device: allocator}

    # ===-------------------------------------------------------------------===#
    # Registration API
    # ===-------------------------------------------------------------------===#

    fn register(mut self, allocator: Some[Allocator]):
        """Register an allocator for its associated device.

        This instance method registers an allocator in this context's registry.
        For global registration, use the static `register()` method.

        Args:
            allocator: The allocator to register.
        """
        self._allocator[allocator.device()] = allocator.getAllocatorImpl()

    @staticmethod
    fn register(impl: Some[Allocator]):
        """Register an allocator globally.

        This static method registers an allocator in the global context's
        registry, making it accessible from anywhere in the program.

        Args:
            impl: The allocator implementation to register.
        """
        try:
            var ctx = Self.get_global_context()
            ctx[].register(impl)
        except err:
            print("Failed to register allocator:", err)
            print(String(err.get_stack_trace()))

    # ===-------------------------------------------------------------------===#
    # Retrieval API
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        ref self, key: Device
    ) raises -> ref [self._allocator[key]] AllocatorImpl:
        return self._allocator[key]

    @always_inline
    fn get_allocator(self, device: Device) -> Optional[AllocatorImpl]:
        """Retrieve an allocator by device from this context.

        Args:
            device: The device to look up.

        Returns:
            The allocator for the device, or None if not registered.
        """
        return self._allocator.get(device)

    @always_inline
    fn get_allocator(
        self, device: Device, default: AllocatorImpl
    ) -> AllocatorImpl:
        """Retrieve an allocator by device with a fallback default.

        Args:
            device: The device to look up.
            default: The allocator to return if device not found.

        Returns:
            The allocator for the device, or the default.
        """
        return self._allocator.get(device, default)

    @always_inline
    fn has_device(self, device: Device) -> Bool:
        """Check if an allocator is registered for the given device.

        Args:
            device: The device to check.
        Returns:
            True if the device has a registered allocator.
        """
        return device in self._allocator

    # ===-------------------------------------------------------------------===#
    # Global Context Access
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn get_global_context() -> UnsafePointer[GlobalContext, MutOrigin.external]:
        return Self.get_or_create_global_context()

    @staticmethod
    fn get_allocator(device: Device) -> Optional[AllocatorImpl]:
        try:
            var ctx = Self.get_global_context()
            return ctx[][device]
        except err:
            print(err, "Allocator for the specified device not available")
            print(String(err.get_stack_trace()))
            return None

    # ===-------------------------------------------------------------------===#
    # Utility Methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn _init_global_context() -> GlobalContext:
        var ctx = GlobalContext(AllocatorImpl.get_default())
        ctx.register(CPUAllocator())
        return ctx^

    @staticmethod
    fn get_or_create_global_context() -> (
        UnsafePointer[GlobalContext, MutOrigin.external]
    ):
        try:
            return GlobalContext._global_ctx_state.get_or_create_ptr()
        except:
            return abort[UnsafePointer[GlobalContext, MutOrigin.external]](
                "Failed to initialize global context"
            )

    @always_inline
    fn __str__(self) -> String:
        return self._allocator.__str__()

    fn debug_info(self) -> String:
        """Get detailed debug information about registered allocators.

        Returns:
            A detailed string representation of the context state.
        """
        var info = String("GlobalContext:\n")
        info += (
            "  Registered allocators: "
            + String(self._allocator.__len__())
            + "\n"
        )
        for device in self._allocator:
            info += "    Device: " + String(device) + "\n"
        return info

    # @staticmethod
    # fn instance() -> ref GlobalContext:
    #     __mlir_op.`lit.global.single_instance`(...) # currently no such op
    # TODO:
    # request this as a new feature from mojo?? is it really required .... decide later
