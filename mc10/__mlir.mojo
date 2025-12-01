from sys.ffi import external_call
from os import abort
from memory import UnsafePointer
from sys import llvm_intrinsic

alias byte = Byte
alias int = Int32
alias long = Int64
alias size_t = UInt64
alias float = Float32
alias double = Float64
alias bool = Scalar[DType.bool]
alias llvm_ptr = __mlir_type.`!llvm.ptr`
alias AnyTrait = type_of(AnyType)


struct AnyStruct[T: AnyType]:
    @implicit
    fn __init__(out self, arg: Self.T):
        ...


fn implements[T2: AnyTrait, T: T2]() -> Bool:
    return True


fn implements[T2: AnyTrait, T: AnyStruct]() -> Bool:
    return False


fn is_trivial[T: AnyTrivialRegType]() -> Bool:
    return True


fn is_trivial[T: AnyStruct]() -> Bool:
    return False


# TODO: Study these magic methods could be used to improve the functionality...

# # "*Re*placement new": destroy the existing SomeHeavy value in the memory,
# # then initialize a new value into the slot.
# __get_address_as_lvalue(somePointer.value) = SomeHeavy(4, 5)

# # Ok to use an lvalue, convert to borrow etc.
# use(__get_address_as_lvalue(somePointer.value))

# # "Placement new": Initialize a new value into uninitialied memory.
# __get_address_as_uninit_lvalue(somePointer.value) = SomeHeavy(4, 5)

# # Error, cannot read from uninitialized memory.
# use(__get_address_as_uninit_lvalue(somePointer.value))

# __get_lvalue_as_address(x)

# # "Placement delete": destroy the initialized object begin pointed to.
# _ = __get_address_as_owned_value(somePointer.value)

# # Result value can be consumed by anything that takes it as an 'owned'
# # argument as well.
# consume(__get_address_as_owned_value(somePointer.value))


@always_inline("nodebug")
fn _tosi8(val: Int8) -> __mlir_type.si8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si8](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tosi16(val: Int16) -> __mlir_type.si16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si16](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tosi32(val: Int32) -> __mlir_type.si32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si32](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tosi64(val: Int64) -> __mlir_type.si64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.si64](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toi8(val: Int8) -> __mlir_type.i8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i8](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toi16(val: Int16) -> __mlir_type.i16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i16](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toi32(val: Int32) -> __mlir_type.i32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i32](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toi64(val: Int64) -> __mlir_type.i64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i64](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toui8(val: UInt8) -> __mlir_type.ui8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toui16(val: UInt16) -> __mlir_type.ui16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui16](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toui32(val: UInt32) -> __mlir_type.ui32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui32](
        val._mlir_value
    )


@always_inline("nodebug")
fn _toui64(val: UInt64) -> __mlir_type.ui64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui64](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tof16(val: Float16) -> __mlir_type.f16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f16](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tof32(val: Float32) -> __mlir_type.f32:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f32](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tof64(val: Float64) -> __mlir_type.f64:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.f64](
        val._mlir_value
    )


@always_inline("nodebug")
fn _tobf16(val: BFloat16) -> __mlir_type.bf16:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.bf16](
        val._mlir_value
    )


@always_inline("nodebug")
fn _fromi8(val: __mlir_type.i8) -> Int8:
    return Int8(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromi32(val: __mlir_type.i32) -> Int32:
    return Int32(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromi64(val: __mlir_type.i64) -> Int64:
    return Int64(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromsi8(val: __mlir_type.si8) -> Int8:
    return Int8(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromsi32(val: __mlir_type.si32) -> Int32:
    return Int32(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromsi64(val: __mlir_type.si64) -> Int64:
    return Int64(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromui8(val: __mlir_type.ui8) -> UInt8:
    return UInt8(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromui32(val: __mlir_type.ui32) -> UInt32:
    return UInt32(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _fromui64(val: __mlir_type.ui64) -> UInt64:
    return UInt64(
        Int(mlir_value=__mlir_op.`index.casts`[_type = __mlir_type.index](val))
    )


@always_inline("nodebug")
fn _tollvmptr(ptr: UnsafePointer) -> __mlir_type.`!llvm.ptr`:
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr`
    ](ptr.address)


@always_inline("nodebug")
fn _fromllvmptr[
    mut: Bool, origin: Origin[mut], //, Type: AnyType
](ptr: __mlir_type.`!llvm.ptr`) -> UnsafePointer[Type, origin]:
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = UnsafePointer[Type, origin]
    ](ptr)


@always_inline("nodebug")
fn _fromllvmptr_to_int64[](ptr: __mlir_type.`!llvm.ptr`) -> __mlir_type.i64:
    return __mlir_op.`llvm.ptrtoint`[_type = __mlir_type.i64](ptr)


@always_inline("nodebug")
fn _fromint64_to_llvmptr(val: __mlir_type.i64) -> __mlir_type.`!llvm.ptr`:
    return __mlir_op.`llvm.inttoptr`[_type = __mlir_type.`!llvm.ptr`](val)


@always_inline("nodebug")
fn _fakeuse[*Type: AnyType](*any: *Type):
    """A no-op function to prevent dead code elimination."""
    return llvm_intrinsic["llvm.fake.use", NoneType._mlir_type](any)
