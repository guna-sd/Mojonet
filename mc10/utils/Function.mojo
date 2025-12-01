from memory import UnsafePointer
from collections import OptionalReg


# I am really not sure about the safeness, but the logic is almost there and it seems working for now...
# Plays an important Role in this framwork...

@register_passable("trivial")
struct Function[FnType: AnyTrivialRegType](
    ImplicitlyBoolable, Stringable, Writable
):
    var _fn: OptionalReg[Self.FnType]
    var _addr: UnsafePointer[Int, ImmutAnyOrigin]

    fn __init__(out self):
        """Create a null Function."""
        self._fn = None
        self._addr = UnsafePointer[Int, ImmutAnyOrigin]()

    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        self = Self()

    @implicit
    fn __init__(out self, function: Self.FnType):
        self._fn = function

        # Take the address of the symbol itself
        var func: UnsafePointer[Int, ImmutAnyOrigin]
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(func))
        UnsafePointer(to=func).bitcast[Self.FnType]()[] = function
        self._addr = func

    fn compare(read self, other: Function[Self.FnType]) -> Bool:
        return self._addr == other._addr

    fn compare_exchange(
        mut self,
        expected: Function[Self.FnType],
        new: Self.FnType,
    ) -> Bool:
        if self._addr == expected._addr:
            self._fn = new
            var func: UnsafePointer[Int, ImmutAnyOrigin]
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(func)
            )
            UnsafePointer(to=func).bitcast[Self.FnType]()[] = new
            self._addr = func
            return True
        return False

    fn compare_exchange(
        mut self,
        expected: Self.FnType,
        new: Self.FnType,
    ) -> Bool:
        if self._fn is not None:
            if self._addr != Function[Self.FnType](expected)._addr:
                return False
            self._fn = new
            var func: UnsafePointer[Int, ImmutAnyOrigin]
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(func)
            )
            UnsafePointer(to=func).bitcast[Self.FnType]()[] = new
            self._addr = func
            return True
        return False

    fn get(read self, /) -> Self.FnType:
        debug_assert(
            self._fn is not None,
            "Attempted to get function from uninitialized Function object.",
        )
        return self._fn.value()

    fn _get_address(read self, /) -> UnsafePointer[Int, ImmutAnyOrigin]:
        return self._addr

    fn __bool__(read self) -> Bool:
        return self._fn is not None

    fn __eq__(read self, other: Function[Self.FnType]) -> Bool:
        return self._addr == other._addr

    fn __ne__(read self, other: Function[Self.FnType]) -> Bool:
        return self._addr != other._addr

    fn __is__(read self, other: Function[Self.FnType]) -> Bool:
        return self._addr == other._addr

    fn __isnot__(read self, other: Function[Self.FnType]) -> Bool:
        return self._addr != other._addr

    fn __is__(self, other: NoneType._mlir_type) -> Bool:
        return not self.__bool__()

    fn __isnot__(self, other: NoneType._mlir_type) -> Bool:
        return self.__bool__()

    fn __int__(read self) -> Int:
        return Int(self._addr)

    fn __str__(read self) -> String:
        return "Function at address: " + String(self._addr)

    fn write_to[W: Writer](read self, mut writer: W):
        writer.write(self._addr)
