from memory import UnsafePointer, ArcPointer
from collections import Optional

from mc10.utils.Symbolic import SymNode, SymNodeImpl
from mc10.utils.SymbolicBool import SymBool
from mc10.utils.SymbolicFloat import SymFloat
from mc10.types.traits import Symbolic
from mc10.__mlir import _toi64
from mc10.utils.debuggable import asserts, abort


# @value
# @register_passable("trivial")
# struct SymInt(Symbolic):
#     alias MAX_UNREPRESENTABLE_INT: Int64 = -1 & ~(1 << 62)
#     alias MASK: UInt64 = (1 << 63) | (1 << 62) | (1 << 61)
#     alias IS_SYM: UInt64 = (1 << 63) | (1 << 61)
#     alias POINTER_MASK = ~(1 << 63 | 1 << 62)

#     var data: Int64

#     fn __init__(out self):
#         self.data = 0

#     fn __init__(out self, value: Int64):
#         if self.is_representable(value):
#             self.data = value
#         else:
#             self.promote(value)

#     fn __init__(out self, node: SymNodeImpl):
#         debug_assert(node.is_int(), "SymNode is not an int")
#         var ptr: UInt64 = node.get_int().value().cast[DType.uint64]()
#         var rep = (ptr & ~Self.MASK) | Self.IS_SYM
#         self.data = rep.cast[DType.int64]()

#     fn __bool__(self) -> Bool:
#         return self.data != 0

#     fn __int__(self) -> Int:
#         if self.is_symbolic():
#             return self.int_to_pointer()[].get_int().value().__int__()
#         return self.data.__int__()

#     fn __add__(self, other: Self) -> Self:
#         return SymInt(self.data + other.data)

#     fn __add__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data + Int(other))

#     fn __add__(self: Self, other: Symbolic) raises -> Self:
#         if other.is_int():
#             return self + other.__int__()
#         raise Error("Incompatible type for addition")

#     fn __sub__(self, other: Self) -> Self:
#         return SymInt(self.data - other.data)

#     fn __sub__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data - Int(other))

#     fn __sub__(self: Self, other: Symbolic) raises -> Self:
#         if other.is_int():
#             return self - other.__int__()
#         raise Error("Incompatible type for subtraction")

#     fn __mul__(self, other: Self) -> Self:
#         return SymInt(self.data * other.data)

#     fn __mul__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data * Int(other))

#     fn __mul__(self: Self, other: Symbolic) raises -> Self:
#         if other.is_int():
#             return self * other.__int__()
#         raise Error("Incompatible type for multiplication")

#     fn __truediv__(self, other: Self) -> Self:
#         return SymInt(self.data.__truediv__(other.data))

#     fn __truediv__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data.__truediv__(Int(other)))

#     fn __truediv__(self: Self, other: Symbolic) raises -> Self:
#         if other.is_int():
#             return self.__truediv__(other.__int__())
#         raise Error("Incompatible type for division")

#     fn __eq__(self, other: Self) -> Bool:
#         return self.data == other.data

#     fn __eq__[T: Intable](self, other: T) -> Bool:
#         return self.data == Int(other)

#     fn __eq__(self: Self, other: Symbolic) raises -> Bool:
#         if other.is_int():
#             return self == other.__int__()
#         return False

#     fn __ne__(self, other: Self) -> Bool:
#         return not self.__eq__(other)

#     fn __ne__[T: Intable](self, other: T) -> Bool:
#         return not self.__eq__(other)

#     fn __ne__(self: Self, other: Symbolic) raises -> Bool:
#         return not self.__eq__(other)

#     fn __lt__(self, other: Self) -> Bool:
#         return self.data < other.data

#     fn __lt__[T: Intable](self, other: T) -> Bool:
#         return self.data < Int(other)

#     fn __lt__(self: Self, other: Symbolic) raises -> Bool:
#         if other.is_int():
#             return self < other.__int__()
#         raise Error("Incompatible type for less-than comparison")

#     fn __le__(self, other: Self) -> Bool:
#         return self.data <= other.data

#     fn __le__[T: Intable](self, other: T) -> Bool:
#         return self.data <= Int(other)

#     fn __le__(self: Self, other: Symbolic) raises -> Bool:
#         if other.is_int():
#             return self <= other.__int__()
#         raise Error("Incompatible type for less-than-or-equal comparison")

#     fn __gt__(self, other: Self) -> Bool:
#         return self.data > other.data

#     fn __gt__[T: Intable](self, other: T) -> Bool:
#         return self.data > Int(other)

#     fn __gt__(self: Self, other: Symbolic) raises -> Bool:
#         if other.is_int():
#             return self > other.__int__()
#         raise Error("Incompatible type for greater-than comparison")

#     fn __ge__(self, other: Self) -> Bool:
#         return self.data >= other.data

#     fn __ge__[T: Intable](self, other: T) -> Bool:
#         return self.data >= Int(other)

#     fn __ge__(self: Self, other: Symbolic) raises -> Bool:
#         if other.is_int():
#             return self >= other.__int__()
#         raise Error("Incompatible type for greater-than-or-equal comparison")

#     fn __str__(self) -> String:
#         return String.write(self.data)

#     fn __repr__(self) -> String:
#         return "SymInt(" + String(self) + ")"

#     fn is_int(self) -> Bool:
#         return True

#     fn is_bool(self) -> Bool:
#         return False

#     fn is_float(self) -> Bool:
#         return False

#     fn get_int(self) raises -> SymInt:
#         return self

#     fn get_bool(self) raises -> SymBool:
#         raise Error("get_bool for SymInt is not supported")

#     fn get_float(self) raises -> SymFloat:
#         raise Error("get_float for SymInt is not supported")

#     fn promote(mut self, value: Int64):
#         ptr = UnsafePointer[SymNodeImpl].alloc(1)
#         ptr.init_pointee_move(SymNodeImpl(value))
#         self.data = Self.pointer_to_int(ptr)

#     @staticmethod
#     fn pointer_to_int(ptr: UnsafePointer[SymNodeImpl]) -> Int64:
#         return Int64((Int(ptr)) | Int(Self.IS_SYM))

#     fn int_to_pointer(read self) -> UnsafePointer[SymNodeImpl]:
#         return __mlir_op.`builtin.unrealized_conversion_cast`[
#             _type = UnsafePointer[SymNodeImpl]._mlir_type
#         ](
#             __mlir_op.`llvm.inttoptr`[_type = UnsafePointer[SymNodeImpl]](
#                 _toi64(Int64(self.data & Self.POINTER_MASK))
#             )
#         )

#     fn is_symbolic(read self) -> Bool:
#         return (Int(self.data) & Int(Self.MASK)) == Int(Self.IS_SYM)

#     @staticmethod
#     fn is_representable(value: Int64) -> Bool:
#         return value > Self.MAX_UNREPRESENTABLE_INT

#     fn is_heap_allocated(self) -> Bool:
#         return not self.is_representable(self.data)

#     fn expect_int(self) raises -> Int64:
#         if not self.is_heap_allocated():
#             return self.data
#         raise Error("Expected an integer but got symbolic")

#     fn clone(self) -> Self:
#         return SymInt(self.data)

# fn toSymNode(self) -> SymNode:
#     debug_assert(self.is_heap_allocated())
#     var unextended_bits: UInt64 = (self.data.cast[DType.uint64]()) & ~Self.MASK
#     var sign_bit_mask: UInt64 = 1 << (62 - 1)
#     var extended_bits: UInt64 = (unextended_bits ^ sign_bit_mask) - sign_bit_mask
#     alias ptr_type = UnsafePointer[UInt]

#     var ptr = UnsafePointer.address_of(extended_bits)
#     var node = SymNode(ptr)
#     return node


@value
@register_passable("trivial")
struct SymInt:
    alias __mlir_type = __mlir_type[
        `!kgen.variant<`,
        Int,
        `, `,
        UnsafePointer[SymNode],
        `>`,
    ]

    var value: Self.__mlir_type

    fn __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__(out self, data: Int):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.__mlir_type, index = Int(0).value
        ](data)

    fn __init__(out self, owned data: UnsafePointer[SymNode]):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.__mlir_type, index = Int(1).value
        ](data)

    fn is_symbolic(self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(1).value](self.value)

    fn __int__(self) -> Int:
        asserts(
            not self.is_symbolic(), "Expected non symbolic, but got symbolic"
        )
        return __mlir_op.`kgen.variant.get`[index = Int(0).value](self.value)

    fn __float__(self) -> Float64:
        asserts(False, "int not a float")
        return Float64()

    fn __bool__(self) -> Bool:
        asserts(
            not self.is_symbolic(), "Expected non symbolic, but got symbolic"
        )
        return __mlir_op.`kgen.variant.get`[index = Int(0).value](
            self.value
        ).__bool__()


# @value
# struct SymInt:
#     var data: Int64
#     var ptr: ArcPointer[UnsafePointer[SymNode]]

#     fn __init__(out self):
#         self.data = 0
#         self.ptr = UnsafePointer[SymNode]()

#     fn __init__(out self, data: Int):
#         self.data = Int(data)
#         self.ptr = UnsafePointer[SymNode]()

#     fn __init__(out self, owned ptr: UnsafePointer[SymNode]):
#         self.data = 0
#         self.ptr = ptr

#     fn __init__(out self, *, other: SymInt):
#         self.data = other.data
#         self.ptr = other.ptr

#     fn is_symbolic(self) -> Bool:
#         return Bool(self.ptr[])

#     fn __bool__(self) -> Bool:
#         return Bool(self.data) or Bool(self.ptr[])

#     fn get_int(self) raises -> SymInt:
#         return self

#     fn get_bool(self) raises -> SymBool:
#         raise Error("get_bool for SymInt is not supported")

#     fn get_float(self) raises -> SymFloat:
#         raise Error("get_float for SymInt is not supported")

#     fn __str__(self) -> String:
#         return String.write(self)

#     fn write_to[W: Writer](self, mut writer: W):
#         try:
#             if self.is_symbolic():
#                 writer.write(self.ptr[][].impl[].get_int().value())
#             writer.write(self.data)
#         except:
#             print("Not initalized")

#     fn __repr__(self) -> String:
#         return "SymInt(" + String(self) + ")"


#     fn __add__(self, other: Self) -> Self:
#         return SymInt(self.data + other.data)

#     fn __add__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data + Int(other))

#     fn __add__(self: Self, other: SymType) raises -> Self:
#         if other.is_int():
#             return self + other.get_int()
#         raise Error("Incompatible type for addition")

#     fn __sub__(self, other: Self) -> Self:
#         return SymInt(self.data - other.data)

#     fn __sub__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data - int(other))

#     fn __sub__(self: Self, other: SymType) raises -> Self:
#         if other.is_int():
#             return self - other.get_int()
#         raise Error("Incompatible type for subtraction")

#     fn __mul__(self, other: Self) -> Self:
#         return SymInt(self.data * other.data)

#     fn __mul__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data * int(other))

#     fn __mul__(self: Self, other: SymType) raises -> Self:
#         if other.is_int():
#             return self * other.get_int()
#         raise Error("Incompatible type for multiplication")

#     fn __truediv__(self, other: Self) -> Self:
#         return SymInt(self.data.__truediv__(other.data))

#     fn __truediv__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data.__truediv__(int(other)))

#     fn __truediv__(self: Self, other: SymType) raises -> Self:
#         if other.is_int():
#             return self.__truediv__(other.get_int())
#         raise Error("Incompatible type for division")

#     fn __eq__(self, other: Self) -> Bool:
#         return self.data == other.data

#     fn __eq__[T: Intable](self, other: T) -> Bool:
#         return self.data == int(other)

#     fn __eq__(self: Self, other: SymType) raises -> Bool:
#         if other.is_int():
#             return self == other.get_int()
#         return False

#     fn __ne__(self, other: Self) -> Bool:
#         return not self.__eq__(other)

#     fn __ne__[T: Intable](self, other: T) -> Bool:
#         return not self.__eq__(other)

#     fn __ne__(self: Self, other: SymType) raises -> Bool:
#         return not self.__eq__(other)

#     fn __lt__(self, other: Self) -> Bool:
#         return self.data < other.data

#     fn __lt__[T: Intable](self, other: T) -> Bool:
#         return self.data < int(other)

#     fn __lt__(self: Self, other: SymType) raises -> Bool:
#         if other.is_int():
#             return self < other.get_int()
#         raise Error("Incompatible type for less-than comparison")

#     fn __le__(self, other: Self) -> Bool:
#         return self.data <= other.data

#     fn __le__[T: Intable](self, other: T) -> Bool:
#         return self.data <= int(other)

#     fn __le__(self: Self, other: SymType) raises -> Bool:
#         if other.is_int():
#             return self <= other.get_int()
#         raise Error("Incompatible type for less-than-or-equal comparison")

#     fn __gt__(self, other: Self) -> Bool:
#         return self.data > other.data

#     fn __gt__[T: Intable](self, other: T) -> Bool:
#         return self.data > int(other)

#     fn __gt__(self: Self, other: SymType) raises -> Bool:
#         if other.is_int():
#             return self > other.get_int()
#         raise Error("Incompatible type for greater-than comparison")

#     fn __ge__(self, other: Self) -> Bool:
#         return self.data >= other.data

#     fn __ge__[T: Intable](self, other: T) -> Bool:
#         return self.data >= int(other)

#     fn __ge__(self: Self, other: SymType) raises -> Bool:
#         if other.is_int():
#             return self >= other.get_int()
#         raise Error("Incompatible type for greater-than-or-equal comparison")

#     fn is_int(self) -> Optional[Bool]:
#         return
#     fn is_bool(self) -> Optional[Bool]:
#         return
#     fn is_float(self) -> Optional[Bool]:
#         return