from mc10.hardware.device import Device
from collections import OptionalReg
from memory import UnsafePointer, OpaquePointer
from mc10.utils.Function import Function
from mc10.utils.unique_ptr import UniquePointer, DeleterFnType, default_delete
from mc10._memory.allocator import Allocator, AllocatorImpl


fn noop_free(mut`_`: OpaquePointer[MutOrigin.external]):
    """No-op deleter."""
    pass


# For now we have two diffrent ptr types (Unique , Data) Pointer, is it really required to have a seperation...??
# TODO: clear the confusion and provide a clean and much more simpler and safer way....

struct DataPointer(Boolable):
    """
    A compact, device-aware smart pointer that owns a block of memory and controls its lifetime through a configurable deleter.
    conceptually similar to `c10::DataPtr`.
    """

    # ---------------------------------------------------------------------- #
    # Compile-time type aliases
    # ---------------------------------------------------------------------- #

    comptime DeleterFn = fn (mut OpaquePointer[MutOrigin.external]) -> None

    var _data: OpaquePointer[MutOrigin.external]
    var _ctx: UniquePointer[NoneType]
    var _device: Device

    # ---------------------------------------------------------------------- #
    # Life cycle
    # ---------------------------------------------------------------------- #

    fn __init__(out self):
        """Create a null DataPointer."""
        self._data = OpaquePointer[MutOrigin.external]()
        self._ctx = UniquePointer[NoneType](noop_free)
        self._device = Device.CPU

    fn __init__(out self, device: Device):
        """Create a null DataPointer."""
        self._data = OpaquePointer[MutOrigin.external]()
        self._ctx = UniquePointer[NoneType](noop_free)
        self._device = device

    fn __init__(out self, data: OpaquePointer[MutOrigin.external]):
        self._data = data
        self._ctx = data
        self._device = Device.CPU

    fn __init__(
        out self,
        data: OpaquePointer[MutOrigin.external],
        ctx: OpaquePointer[MutOrigin.external],
        deleter: Self.DeleterFn,
        device: Device,
    ):
        self._data = data
        self._ctx = UniquePointer[NoneType](ctx, deleter)
        self._device = device

    @always_inline
    fn __moveinit__(out self, deinit existing: Self):
        """
        Move constructor. Takes ownership from `existing`.
        `existing` is consumed and will NOT call its deleter.
        """
        self._data = existing._data
        self._ctx = existing._ctx^
        self._device = existing._device

    fn __bool__(read self) -> Bool:
        return (
            self._data != OpaquePointer[MutOrigin.external]()
            or self._ctx != None
        )

    fn __eq__(read self, other: None) -> Bool:
        return not self.__bool__()

    fn __ne__(read self, other: None) -> Bool:
        return self.__bool__()

    # ---------------------------------------------------------------------- #
    # Operations
    # ---------------------------------------------------------------------- #

    fn clear(mut self):
        """Reset to null, without calling deleter."""
        self._data = OpaquePointer[MutOrigin.external]()
        self._ctx.clear()

    fn unsafe_reset_data_and_ctx(
        mut self,
        new_data_and_ctx: OpaquePointer[MutOrigin.external],
    ) -> Bool:
        if self._ctx.get_deleter() is not noop_free:
            return False

        _ = self._ctx.release()
        self._ctx.reset(new_data_and_ctx)
        self._data = new_data_and_ctx
        return True

    fn release_context(mut self) -> OpaquePointer[MutOrigin.external]:
        """Relinquish ownership. Returns ctx, clears internal state."""
        return self._ctx.release()

    fn move_context(mut self) -> UniquePointer[NoneType]:
        """Relinquish ownership. Returns ctx, clears internal state."""
        var ctx = self._ctx^
        self._ctx = UniquePointer[NoneType](ctx.get_deleter())
        return ctx^

    fn get_context(self) -> OpaquePointer[MutOrigin.external]:
        return self._ctx.get()

    fn get_deleter(self) -> DeleterFnType[NoneType]:
        return self._ctx.get_deleter()

    fn compare_exchange_deleter(
        mut self,
        expected: Self.DeleterFn,
        new_deleter: Self.DeleterFn,
    ) -> Bool:
        return self._ctx.compare_exchange(expected, new_deleter)

    fn device(self) -> Device:
        return self._device

    fn get(mut self) -> OpaquePointer[MutOrigin.external]:
        return self._data

    fn unsafe_ptr[
        mut: Bool,
        origin: Origin[mut], //,
        Type: AnyType,
    ](ref [origin]self) -> UnsafePointer[Type, origin]:
        return (
            self._data.bitcast[Type]()
            .mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )

    fn unsafe_set_device(mut self, device: Device):
        self._device = device

    fn cast_context[
        T: AnyType
    ](self, expected_deleter: DeleterFnType[NoneType]) -> UnsafePointer[
        T, MutOrigin.external
    ]:
        if self.get_deleter() != expected_deleter:
            return UnsafePointer[T, MutOrigin.external]()
        return self._ctx.get().bitcast[T]()
