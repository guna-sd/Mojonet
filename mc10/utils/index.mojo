from mc10.__mlir import _toi64, MlirType
from builtin.debug_assert import debug_assert as asserts
from os import abort
from math import Ceilable, CeilDivable, Floorable, Truncable
from utils._select import _select_register_value


@always_inline
fn indexable[
    Type: Sized, //, name: StringLiteral
](idx: index, container: Type) -> index:
    """
    Ensures `idx` is within bounds and returns the indexed value.

    Args:
        idx : The index to access.
        container : A sized container (array, list, etc.).

    Returns:
        Indexable in the container.
    """

    asserts(len(container) >= 0, "Container must have a valid size!")

    asserts(
        len(container) > 0,
        "Attempting to index into an empty ",
        name,
        " container with 0 elements!",
    )
    if idx < 0:
        print(String("Index must be non-negative! Got idx: ") + idx.__str__())
        abort()

    if idx >= len(container):
        print(
            String("Index out of bounds! Provided idx: ")
            + idx.__str__()
            + String(" is larger than container length: ")
            + len(container).__str__()
        )
        abort()

    return idx


@always_inline
fn indexable[
    Type: Sized, //, name: StringLiteral
](idx: Int, container: Type) -> Int:
    """
    Ensures `idx` is within bounds and returns the indexed value.

    Args:
        idx : The index to access.
        container : A sized container (array, list, etc.).

    Returns:
        Indexable in the container.
    """

    asserts(len(container) >= 0, "Container must have a valid size!")

    asserts(
        len(container) > 0,
        "Attempting to index into an empty ",
        name,
        " container with 0 elements!",
    )
    if idx < 0:
        print(String("Index must be non-negative! Got idx: ") + idx.__str__())
        abort()
    if idx >= len(container):
        print(
            String("Index out of bounds! Provided idx: ")
            + idx.__str__()
            + String(" is larger than container length: ")
            + len(container).__str__()
        )
        abort()

    return idx


@register_passable("trivial")
struct index(
    Absable,
    Ceilable,
    CeilDivable,
    Comparable,
    ExplicitlyCopyable,
    Floorable,
    ImplicitlyBoolable,
    KeyElement,
    MlirType,
    Roundable,
    Stringable,
    Truncable,
    Writable,
):
    alias Type = __mlir_type.index
    alias elem_type = DType.index

    var value: Self.Type

    @always_inline("builtin")
    fn __init__(out self):
        self.value = __mlir_attr.`0:index`

    @always_inline("builtin")
    fn __init__(out self: Self, *, other: Self):
        self.value = other.value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.i64):
        self.value = __mlir_op.`index.casts`[_type = __mlir_type.index](value)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        self.value = value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<index>`):
        self = Self(Int(value))

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<i64>`):
        self = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.index](value)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral):
        self = Self(Int(value))

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int64):
        self.value = value.__int__().__index__()

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int32):
        self = value.__index__()

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Int):
        self = value.value

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: UInt):
        self = value.value

    @always_inline("builtin")
    fn __index__(self) -> Self.Type:
        return self.value

    @always_inline("builtin")
    fn __eq__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __ne__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __gt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __ge__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sge>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __lt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate slt>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __le__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sle>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __pos__(self) -> Self:
        return self

    @always_inline("builtin")
    fn __neg__(self) -> Self:
        return self * index(__mlir_attr.`-1:index`)

    @always_inline("builtin")
    fn __add__(self, rhs: Self) -> Self:
        return __mlir_op.`index.add`(self.value, rhs.value)

    @always_inline("builtin")
    fn __sub__(self, rhs: Self) -> Self:
        return __mlir_op.`index.sub`(self.value, rhs.value)

    @always_inline("builtin")
    fn __mul__(self, rhs: Self) -> Self:
        return __mlir_op.`index.mul`(self.value, rhs.value)

    @always_inline("builtin")
    fn __truediv__(self, rhs: Self) -> Self:
        return __mlir_op.`index.divs`(self.value, rhs.value)

    @always_inline("builtin")
    fn __floordiv__(self, rhs: Self) -> Self:
        return __mlir_op.`index.floordivs`(self.value, rhs.value)

    @always_inline("builtin")
    fn __mod__(self, rhs: Self) -> Self:
        return __mlir_op.`index.rems`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __divmod__(self, rhs: Int) -> Tuple[Self, Self]:
        return (self // rhs, self % rhs)

    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        if exp < 0:
            abort("Invalid negative exponent not supported!")
        var res: Self = 1
        var x = self
        var n = exp
        while n > 0:
            if n & 1 != 0:
                res *= x
            x *= x
            n >>= 1
        return res

    @always_inline("builtin")
    fn __lshift__(self, rhs: Self) -> Self:
        """Return `self << rhs`."""
        return _select_register_value(
            rhs < 0, 0, __mlir_op.`index.shl`(self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __rshift__(self, rhs: Self) -> Self:
        """Return `self >> rhs`."""
        return _select_register_value(
            rhs < 0, 0, __mlir_op.`index.shrs`(self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __and__(self, rhs: Self) -> Self:
        return __mlir_op.`index.and`(self.value, rhs.value)

    @always_inline("builtin")
    fn __or__(self, rhs: Self) -> Self:
        return __mlir_op.`index.or`(self.value, rhs.value)

    @always_inline("builtin")
    fn __xor__(self, rhs: Self) -> Self:
        return __mlir_op.`index.xor`(self.value, rhs.value)

    @always_inline("builtin")
    fn __invert__(self) -> Self:
        return self ^ index(__mlir_attr.`-1:index`)

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Self):
        self = self / rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Self):
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(mut self, rhs: Self):
        self = self % rhs

    @always_inline("nodebug")
    fn __ipow__(mut self, rhs: Self):
        self = self**rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: Self):
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Self):
        self = self >> rhs

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Self):
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Self):
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Self):
        self = self | rhs

    @always_inline("builtin")
    fn __radd__(self, value: Self) -> Self:
        return self + value

    @always_inline("builtin")
    fn __rsub__(self, value: Self) -> Self:
        return value - self

    @always_inline("builtin")
    fn __rmul__(self, value: Self) -> Self:
        return self * value

    @always_inline("builtin")
    fn __rfloordiv__(self, value: Self) -> Self:
        return value // self

    @always_inline("builtin")
    fn __rmod__(self, value: Self) -> Self:
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, value: Self) -> Self:
        return value**self

    @always_inline("builtin")
    fn __rlshift__(self, value: Self) -> Self:
        return value << self

    @always_inline("builtin")
    fn __rrshift__(self, value: Self) -> Self:
        return value >> self

    @always_inline("builtin")
    fn __rand__(self, value: Self) -> Self:
        return value & self

    @always_inline("builtin")
    fn __ror__(self, value: Self) -> Self:
        return value | self

    @always_inline("builtin")
    fn __rxor__(self, value: Self) -> Self:
        return value ^ self

    @always_inline("builtin")
    fn __abs__(self) -> Self:
        return _select_register_value(self < 0, -self, self)

    @always_inline("builtin")
    fn __ceil__(self) -> Self:
        return self

    @always_inline("builtin")
    fn __floor__(self) -> Self:
        return self

    @always_inline("builtin")
    fn __round__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        return _select_register_value(
            ndigits >= 0, self, self - (self % 10 ** -(ndigits))
        )

    @always_inline("builtin")
    fn __trunc__(self) -> Self:
        return self

    @always_inline("builtin")
    fn __ceildiv__(self, denominator: Self) -> Self:
        return -(self // -denominator)

    @always_inline("builtin")
    fn is_power_of_two(self) -> Bool:
        """Check if the integer is a (non-zero) power of two.

        Returns:
            True if the integer is a power of two, False otherwise.
        """
        return (self & (self - 1) == 0) & (self > 0)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return self != 0

    @always_inline("builtin")
    fn __as_bool__(self) -> Bool:
        return self.__bool__()

    @always_inline("builtin")
    fn __int__(self) -> Int:
        return Int(self.value)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__int__())

    fn __hash__(self) -> UInt:
        return self.__int__().__hash__()
