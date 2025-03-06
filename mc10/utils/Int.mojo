from mc10.__mlir import _toi32, MlirType
from mc10.utils.debuggable import abort
from math import Ceilable, CeilDivable, Floorable, Truncable


@value
@register_passable("trivial")
struct int(
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
):
    alias MAX = Self(Scalar[DType.int32].MAX)
    alias MIN = Self(Scalar[DType.int32].MIN)
    alias Type = __mlir_type.i32
    alias elem_type = DType.int32

    var value: Self.Type

    @always_inline("nodebug")
    fn __init__(out self):
        self.value = __mlir_attr.`0:i32`

    @always_inline("nodebug")
    fn __init__(out self: Self, *, other: Self):
        self.value = other.value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.i32):
        self.value = value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        self.value = __mlir_op.`index.casts`[_type = __mlir_type.i32](value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<index>`):
        self = Self(Int(value))

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<i32>`):
        self = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i32](value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<si32>`):
        self = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i32](value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: IntLiteral):
        self = Self(Int(value))

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int32):
        self.value = _toi32(value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        self = Self(value.value)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: UInt):
        self = Self(value.value)

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        return __mlir_op.`index.casts`[_type = __mlir_type.index](self.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self.__mlir_index__(), rhs.__mlir_index__())
        # return __mlir_op.`arith.cmpi`[_type = __mlir_type.i1, predicate = __mlir_attr.`0`](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sge>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate slt>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sle>`
        ](self.__mlir_index__(), rhs.__mlir_index__())

    @always_inline("nodebug")
    fn __pos__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __neg__(self) -> Self:
        return self * -1

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        # return __mlir_op.`index.add`(
        #     self.__mlir_index__(), rhs.__mlir_index__()
        # )
        return __mlir_op.`llvm.add`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.sub`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.mul`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.sdiv`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        return __mlir_op.`index.floordivs`(
            self.__mlir_index__(), rhs.__mlir_index__()
        )

    @always_inline("nodebug")
    fn __mod__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.srem`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

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

    @always_inline("nodebug")
    fn __lshift__(self, rhs: Self) -> Self:
        """Return `self << rhs`."""
        if rhs < 0:
            abort("Shift cannot be negative.")
        return __mlir_op.`llvm.shl`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __rshift__(self, rhs: Self) -> Self:
        """Return `self >> rhs`."""
        if rhs < 0:
            abort("Shift cannot be negative.")
        return __mlir_op.`llvm.shr`[_type = __mlir_type.i32](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __and__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __or__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.or`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __xor__(self, rhs: Self) -> Self:
        return __mlir_op.`llvm.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __invert__(self) -> Self:
        return self ^ -1

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        self = self * rhs

    fn __itruediv__(mut self, rhs: Self):
        self = self // rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Self):
        self = self // rhs

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

    @always_inline("nodebug")
    fn __radd__(self, value: Self) -> Self:
        return self + value

    @always_inline("nodebug")
    fn __rsub__(self, value: Self) -> Self:
        return value - self

    @always_inline("nodebug")
    fn __rmul__(self, value: Self) -> Self:
        return self * value

    @always_inline("nodebug")
    fn __rfloordiv__(self, value: Self) -> Self:
        return value // self

    @always_inline("nodebug")
    fn __rmod__(self, value: Self) -> Self:
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, value: Self) -> Self:
        return value**self

    @always_inline("nodebug")
    fn __rlshift__(self, value: Self) -> Self:
        return value << self

    @always_inline("nodebug")
    fn __rrshift__(self, value: Self) -> Self:
        return value >> self

    @always_inline("nodebug")
    fn __rand__(self, value: Self) -> Self:
        return value & self

    @always_inline("nodebug")
    fn __ror__(self, value: Self) -> Self:
        return value | self

    @always_inline("nodebug")
    fn __rxor__(self, value: Self) -> Self:
        return value ^ self

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return self != 0

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        return self.__bool__()

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return Int(self.__mlir_index__())

    @always_inline("nodebug")
    fn __abs__(self) -> Self:
        return self.__int__().__abs__()

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __round__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        if ndigits >= 0:
            return self
        return self - (self % 10 ** -(ndigits))

    @always_inline("nodebug")
    fn __trunc__(self) -> Self:
        return self

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        return -(self // -denominator)

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
