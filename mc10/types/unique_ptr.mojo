from mc10.utils.Function import Function
from collections.string.string_slice import _get_kgen_string

comptime DeleterFnType[Type: AnyType] = Function[
    fn (mut ptr: UnsafePointer[Type, MutOrigin.external])
]


fn default_delete[
    Type: AnyType,
](mut ptr: UnsafePointer[Type, MutOrigin.external]):
    """Default deleter function that frees memory allocated for Self.Type."""
    ptr.free()


# Considered as Utility required by DataPointer which is a core type used in the framework Yet to decide the requirements...
# # TODO: clear the confusion and provide a clean and much more simpler and safer way....


@register_passable
struct UniquePointer[
    Type: AnyType,
](Boolable, Movable):
    var _ptr: UnsafePointer[Self.Type, MutOrigin.external]
    var _deleter: DeleterFnType[Self.Type]

    fn __init__(out self):
        """Create a null UniquePointer."""
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()
        self._deleter = DeleterFnType[Self.Type](default_delete[Self.Type])

    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        """Create a null UniquePointer."""
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()
        self._deleter = DeleterFnType[Self.Type](default_delete[Self.Type])

    @implicit
    fn __init__(
        out self,
        ptr: UnsafePointer[Self.Type, MutOrigin.external],
    ):
        self._ptr = ptr
        self._deleter = DeleterFnType[Self.Type](default_delete[Self.Type])

    fn __init__(
        out self,
        deleter: DeleterFnType[Self.Type],
    ):
        """Create a null UniquePointer with custom deleter."""
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()
        self._deleter = deleter

    fn __init__(
        out self,
        ptr: UnsafePointer[Self.Type, MutOrigin.external],
        deleter: DeleterFnType[Self.Type] = DeleterFnType[Self.Type](
            default_delete[Self.Type]
        ),
    ):
        self._ptr = ptr
        self._deleter = deleter

    @always_inline
    fn __copyinit__(out self, existing: Self):
        """Disable copy constructor."""
        comptime msg = "Attempted to copy a UniquePointer."
        constrained[False, msg]()
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()
        self._deleter = DeleterFnType[Self.Type](default_delete[Self.Type])
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __del__(deinit self):
        if self._ptr != UnsafePointer[Self.Type, MutOrigin.external]():
            if self._deleter:
                self._deleter.get()(self._ptr)
            else:
                # No deleter set; leak the memory
                pass

    fn __bool__(read self) -> Bool:
        return self._ptr != UnsafePointer[Self.Type, MutOrigin.external]()

    fn __eq__(read self, other: None) -> Bool:
        return not self.__bool__()

    fn __ne__(read self, other: None) -> Bool:
        return self.__bool__()

    fn __getitem__(
        ref [AddressSpace.GENERIC]self,
    ) -> ref [self, AddressSpace.GENERIC] Self.Type:
        """Returns a reference to the pointers's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the `OwnedPointer`.
        """
        # This should have a widening conversion here that allows
        # the mutable ref that is always (potentially unsafely)
        # returned from UnsafePointer to be guarded behind the
        # aliasing guarantees of the origin system here.
        # All of the magic happens above in the function signature
        return self._ptr[]

    fn get(read self) -> UnsafePointer[Self.Type, MutOrigin.external]:
        return self._ptr

    fn unsafe_ptr[
        mut: Bool,
        origin: Origin[mut], //,
    ](ref [origin]self) -> UnsafePointer[Self.Type, origin]:
        """Returns the backing pointer for this `OwnedPointer`.

        Parameters:
            mut: Whether the pointer is mutable.
            origin: The origin of the pointer.

        Returns:
            An UnsafePointer to the backing allocation for this `OwnedPointer`.
        """
        return self._ptr.mut_cast[mut]().unsafe_origin_cast[origin]()

    fn release(mut self) -> UnsafePointer[Self.Type, MutOrigin.external]:
        var temp = self._ptr
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()
        return temp

    fn clear(mut self):
        """Reset to null, without calling deleter."""
        self._ptr = UnsafePointer[Self.Type, MutOrigin.external]()

    fn reset(mut self, new_ptr: UnsafePointer[Self.Type, MutOrigin.external]):
        if new_ptr == self._ptr:
            # no-op, like std::unique_ptr...
            return

        if self._ptr != UnsafePointer[Self.Type, MutOrigin.external]():
            self._deleter.get()(self._ptr)

        self._ptr = new_ptr

    fn get_deleter(read self) -> DeleterFnType[Self.Type]:
        return self._deleter

    fn swap(mut self, mut other: UniquePointer[Self.Type]):
        var temp = self._ptr
        self._ptr = other._ptr
        other._ptr = temp
        var temp_deleter = self._deleter
        self._deleter = other._deleter
        other._deleter = temp_deleter

    fn compare_exchange(
        mut self,
        expected: DeleterFnType[Self.Type].FnType,
        new_deleter: DeleterFnType[Self.Type].FnType,
    ) -> Bool:
        return self._deleter.compare_exchange(expected, new_deleter)
