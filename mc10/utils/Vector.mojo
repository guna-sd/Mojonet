from _mlir.builtin_types import DialectType, FunctionType, MLIR_func, Context, Type
from _mlir.builtin_attributes import Attribute, BoolAttr, TypeAttr, StringAttr, DialectAttribute, BuiltinAttributes, BuiltinTypes
from _mlir.diagnostics import DiagnosticSeverity, MlirLogicalResult
from _mlir.ir import NamedAttribute, _WriteState, Dialect, DialectHandle, DialectRegistry, Location, Operation, _OpBuilderList, IR, Region, Value, Block, Module
from _mlir.rewrite import Rewriter
from sys.ffi import _Global, _mlirtype_is_eq

alias a = BuiltinTypes.mlirContextCreate()
alias f162 = (BuiltinTypes.mlirF16TypeGet(a))
alias asa = Type(f162)

struct DataType:
    alias typecontext = BuiltinTypes.mlirContextCreate()
    alias i8 = Self(BuiltinTypes.mlirIntegerTypeGet(Self.typecontext, 8))
    alias i16 = Self(BuiltinTypes.mlirIntegerTypeGet(Self.typecontext, 16))
    alias i32 = Self(BuiltinTypes.mlirIntegerTypeGet(Self.typecontext, 32))
    alias i64 = Self(BuiltinTypes.mlirIntegerTypeGet(Self.typecontext, 64))

    alias si8 = Self(BuiltinTypes.mlirIntegerTypeSignedGet(Self.typecontext, 8))
    alias si16 = Self(BuiltinTypes.mlirIntegerTypeSignedGet(Self.typecontext, 16))
    alias si32 = Self(BuiltinTypes.mlirIntegerTypeSignedGet(Self.typecontext, 32))
    alias si64 = Self(BuiltinTypes.mlirIntegerTypeSignedGet(Self.typecontext, 64))

    alias ui8 = Self(BuiltinTypes.mlirIntegerTypeUnsignedGet(Self.typecontext, 8))
    alias ui16 = Self(BuiltinTypes.mlirIntegerTypeUnsignedGet(Self.typecontext, 16))
    alias ui32 = Self(BuiltinTypes.mlirIntegerTypeUnsignedGet(Self.typecontext, 32))
    alias ui64 = Self(BuiltinTypes.mlirIntegerTypeUnsignedGet(Self.typecontext, 64))

    alias f16 = Self(BuiltinTypes.mlirF16TypeGet(Self.typecontext))
    alias f32 = Self(BuiltinTypes.mlirF32TypeGet(Self.typecontext))
    alias f64 = Self(BuiltinTypes.mlirF64TypeGet(Self.typecontext))

    alias bf16 = Self(BuiltinTypes.mlirBF16TypeGet(Self.typecontext))

    var type : Type

    @implicit
    fn __init__(out self, type: BuiltinTypes.MlirType):
        self.type = Type(type)
    

struct DataTypes(AnyType):
    alias i8 = 0
    alias i16 = Self(__mlir_type.i16)
    alias i32 = Self(__mlir_type.i32)
    alias i64 = Self(__mlir_type.i64)

    alias si8 = Self( __mlir_type.si8)
    alias si16 = Self(__mlir_type.si16)
    alias si32 = Self(__mlir_type.si32)
    alias si64 = Self(__mlir_type.si64)

    alias ui8 = Self(__mlir_type.ui8)
    alias ui16 = Self(__mlir_type.ui16)
    alias ui32 = Self(__mlir_type.ui32)
    alias ui64 = Self(__mlir_type.ui64)

    alias f16 = Self(__mlir_type.f16)
    alias f32 = Self(__mlir_type.f32)
    alias f64 = Self(__mlir_type.f64)
    alias bf16 = Self(__mlir_type.bf16)

    alias none = Self(__mlir_type.`none`)

    var value: AnyTrivialRegType

    @implicit
    fn __init__(out self, value: AnyTrivialRegType):
        self.value = value

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        return "MLIRType." + String(self)
    
    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if _mlirtype_is_eq[__type_of(self.value), __type_of(Self.i8.value)]():
            writer.write("i8")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.i16.value)]():
            writer.write("i16")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.i32.value)]():
            writer.write("i32")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.i64.value)]():
            writer.write("i64")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.si8.value)]():
            writer.write("si8")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.si16.value)]():
            writer.write("si16")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.si32.value)]():
            writer.write("si32")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.si64.value)]():
            writer.write("si64")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.ui8.value)]():
            writer.write("ui8")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.ui16.value)]():
            writer.write("ui16")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.ui32.value)]():
            writer.write("ui32")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.ui64.value)]():
            writer.write("ui64")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.f16.value)]():
            writer.write("f16")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.f32.value)]():
            writer.write("f32")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.f64.value)]():
            writer.write("f64")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.bf16.value)]():
            writer.write("bf16")
        elif _mlirtype_is_eq[__type_of(self.value), __type_of(Self.none.value)]():
            writer.write("none")
        else:
            writer.write("Unknown type")

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        if _mlirtype_is_eq[__type_of(self.value), __type_of(rhs.value)]():
            return True
        return False

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        return not(self == rhs)

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        return self != rhs

# var I1 = __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i1, value=__mlir_attr[`0:i1`]]()
# var I8 = __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i8, value=__mlir_attr[`0:i8`]]()
# var I16 = __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i16, value=__mlir_attr[`0:i16`]]()
# var I32 = __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i32, value=__mlir_attr[`0:i32`]]()
# var I64 = __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i64, value=__mlir_attr[`0:i64`]]()


# struct DataTypes:
#     alias i1 = __type_of(I1)
#     alias i8 = __type_of(I8)
#     alias i16 = __type_of(I16)
#     alias i32 = __type_of(I32)
#     alias i64 = __type_of(I64)
