from memory import UnsafePointer, ArcPointer
from collections import Optional

from max.driver.anytensor import AnyTensor, AnyMemory, AnyMojoValue, _CMojoValue
from max.graph.symbol import Symbol
from max.graph.type import Type, TensorType
from max.graph._attributes import _shape_attr, _scalar_attr
from max.graph.graph import _vector_attr
from max.graph.graph import _OwnedGraph, Operation

from mc10.utils.Symbolic import SymNode
from mc10.utils.SymbolicBool import SymBool
from mc10.utils.SymbolicFloat import SymFloat
# @value
# struct SymInt(SymType):
#     alias MAX_UNREPRESENTABLE_INT: Int64 = -1 & ~(1 << 62)
#     alias MASK: UInt64 = (1 << 63) | (1 << 62) | (1 << 61)
#     alias IS_SYM: UInt64 = (1 << 63) | (1 << 61)
    
#     var data: Int64

#     fn __init__(inout self):
#         self.data = 0

#     fn __init__(inout self, value: Int64):
#         self.data = value
#         if self.is_heap_allocated():
#             self.promote_to_negative()
    
#     fn __init__(inout self, node: SymNode):
#         debug_assert(node.is_int(), "SymNode is not an int")
#         var ptr: UInt64 = node.value.bitcast[UInt64]()[]
#         var rep = (ptr & ~Self.MASK) | Self.IS_SYM
#         self.data = rep.cast[DType.int64]()

#     fn __bool__(self) -> Bool:
#         return self.data != 0

#     fn __add__(self, other: Self) -> Self:
#         return SymInt(self.data + other.data)

#     fn __add__[T: Intable](self, other: T) -> Self:
#         return SymInt(self.data + int(other))

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

#     fn __str__(self) -> String:
#         return String.write(self.data)

#     fn __repr__(self) -> String:
#         return "SymInt(" + str(self) + ")"

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

#     fn promote_to_negative(inout self):
#         # var node = SymNode(SymNodeType.Int, Arc((self.data)))
#         # var sym_int = SymInt(node)
#         # self.data = sym_int.data
#         # sym_int.data = 0
#         ...

#     @staticmethod
#     fn check(value: Int64) -> Bool:
#         return value > Self.MAX_UNREPRESENTABLE_INT

#     fn is_heap_allocated(self) -> Bool:
#         return not self.check(self.data)

#     fn is_symbolic(self) -> Bool:
#         return self.is_heap_allocated() and self.toSymNode().

#     fn constant_int(self) raises -> Int64:
#         if self.is_symbolic():
#             raise Error("This is a symbolic value, not a constant.")
#         return self.data

#     fn expect_int(self) raises -> Int64:
#         if not self.is_heap_allocated():
#             return self.data
#         raise Error("Expected an integer but got symbolic")

#     fn clone(self) -> Self:
#         return SymInt(self.data)
    
#     fn toSymNode(self) -> SymNode:
#         debug_assert(self.is_heap_allocated())
#         var unextended_bits: UInt64 = (self.data.cast[DType.uint64]()) & ~Self.MASK
#         var sign_bit_mask: UInt64 = 1 << (62 - 1)
#         var extended_bits: UInt64 = (unextended_bits ^ sign_bit_mask) - sign_bit_mask
#         alias ptr_type = UnsafePointer[UInt]

#         var ptr = UnsafePointer.address_of(extended_bits).bitcast[UnsafePointer[UInt]]()
#         var node = SymNode(SymNodeType.Int, ptr)
#         return node

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