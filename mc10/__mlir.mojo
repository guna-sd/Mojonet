from sys.intrinsics import _mlirtype_is_eq, _type_is_eq
from sys.ffi import _get_global_or_null, external_call
from os import abort
from memory import UnsafePointer
from _mlir._c.BuiltinTypes import (
    MlirContext,
    mlirContextCreate,
    mlirContextDestroy,
)


fn newRuntime() -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_AsyncRT_CreateRuntime", UnsafePointer[NoneType]
    ](0)


fn getRuntime() -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime", UnsafePointer[NoneType]
    ]()


fn delRuntime(ptr: UnsafePointer[NoneType]):
    external_call["KGEN_CompilerRT_AsyncRT_DestroyRuntime", NoneType](ptr)


fn newMlirContext() -> MlirContext:
    return mlirContextCreate()


fn delMlirContext(ctx: MlirContext):
    mlirContextDestroy(ctx)


alias byte = Byte
alias int = Int32
alias long = Int64
alias size_t = UInt64
alias float = Float32
alias double = Float64
alias bool = Scalar[DType.bool]
alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


struct AnyStruct[T: AnyType]:
    @implicit
    fn __init__(out self, arg: T):
        ...


fn implements[T2: _AnyTypeMetaType, T: T2]() -> Bool:
    return True


fn implements[T2: _AnyTypeMetaType, T: AnyStruct]() -> Bool:
    return False


fn is_trivial[T: AnyTrivialRegType]() -> Bool:
    return True


fn is_trivial[T: AnyStruct]() -> Bool:
    return False


fn _tosi8(val: Int8) -> __mlir_type.si8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si8](val.value)


fn _tosi16(val: Int16) -> __mlir_type.si16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si16](val.value)


fn _tosi32(val: Int32) -> __mlir_type.si32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si32](val.value)


fn _tosi64(val: Int64) -> __mlir_type.si64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si64](val.value)


fn _toi8(val: Int8) -> __mlir_type.i8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i8](val.value)


fn _toi16(val: Int16) -> __mlir_type.i16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i16](val.value)


fn _toi32(val: Int32) -> __mlir_type.i32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i32](val.value)


fn _toi64(val: Int64) -> __mlir_type.i64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i64](val.value)


fn _toui8(val: UInt8) -> __mlir_type.ui8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](val.value)


fn _toui16(val: UInt16) -> __mlir_type.ui16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui16](val.value)


fn _toui32(val: UInt32) -> __mlir_type.ui32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui32](val.value)


fn _toui64(val: UInt64) -> __mlir_type.ui64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui64](val.value)


fn _tof16(val: Float16) -> __mlir_type.f16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f16](val.value)


fn _tof32(val: Float32) -> __mlir_type.f32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f32](val.value)


fn _tof64(val: Float64) -> __mlir_type.f64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f64](val.value)


fn _tobf16(val: BFloat16) -> __mlir_type.bf16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.bf16](val.value)


fn _fromi8(val: __mlir_type.i8) -> Int8:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromi32(val: __mlir_type.i32) -> Int32:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromi64(val: __mlir_type.i64) -> Int64:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromsi8(val: __mlir_type.si8) -> Int8:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromsi32(val: __mlir_type.si32) -> Int32:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromsi64(val: __mlir_type.si64) -> Int64:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromui8(val: __mlir_type.ui8) -> UInt8:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromui32(val: __mlir_type.ui32) -> UInt32:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn _fromui64(val: __mlir_type.ui64) -> UInt64:
    return __mlir_op.`index.casts`[_type = __mlir_type.index](val)


fn vector1d(val: Int32, out vector: __mlir_type.`vector<1xi32>`):
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(vector))
    return __mlir_op.`llvm.insertelement`[_type = __type_of(vector)](
        vector, _toi32(val), _toi32(0)
    )


trait MlirType(CollectionElement):
    alias Type: AnyTrivialRegType
    alias elem_type: DType


@register_passable("trivial")
struct i8(MlirType):
    alias Type = __mlir_type.i8
    alias elem_type = DType.int8


@register_passable("trivial")
struct i16(MlirType):
    alias Type = __mlir_type.i16
    alias elem_type = DType.int16


@register_passable("trivial")
struct i32(MlirType):
    alias Type = __mlir_type.i32
    alias elem_type = DType.int32


@register_passable("trivial")
struct i64(MlirType):
    alias Type = __mlir_type.i64
    alias elem_type = DType.int64


alias value = __mlir_type[
    `!pop.union<`, __mlir_type.i64, `,`, __mlir_type.f64, `>`
]
