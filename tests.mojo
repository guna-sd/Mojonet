# # from sys import llvm_intrinsic, is_gpu, bitwidthof
# # from sys.ffi import external_call
# # from sys.intrinsics import _mlirtype_is_eq
# # from memory import UnsafePointer

# # from compile.compile import compile_info

# # def _malloc[Type: AnyTrivialRegType](size: Int) -> __mlir_type.`!llvm.ptr<0>` as ptr:
# #     return external_call["malloc", __type_of(ptr)](size)


# # def __fma[dtype: DType](a: Scalar[dtype], b:  Scalar[dtype], c: Scalar[dtype]) -> Scalar[dtype]:
# #     if dtype is DType.float16:
# #         if dtype is DType.float16:
# #             return llvm_intrinsic["llvm.fma.f16", Scalar[dtype], has_side_effect=False](a,b,c)
# #         if dtype is DType.float32:
# #             return llvm_intrinsic["llvm.fma.f32", Scalar[dtype], has_side_effect=False](a,b,c)
# #         if dtype is DType.float64:
# #             return llvm_intrinsic["llvm.fma.f64", Scalar[dtype], has_side_effect=False](a,b,c)
# #     return Scalar[dtype](a * b + c)

# # def a():
# #     var i = 231
# #     print(i)

# # def addveckernel[threadIdx: Int, blockIdx: Int, blockDim: Int](A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], n:Int):
# #     var tid = threadIdx
# #     var bid = blockIdx
# #     var block_size = blockDim

# #     var global_idx = tid + bid * block_size

# #     if global_idx < n:
# #         var a_val = A[global_idx]
# #         var b_val = B[global_idx]
# #         var c_val = a_val + b_val

# #         C[global_idx] = c_val
# #     return

# # from gpu.host._compile import _ptxas_compile, _compile_code_asm, get_linkage_name
# # from max.graph import Symbol
# # from max.engine import Value
# # from max.engine.value import CValue
# # def ts():
# #     print("hii")


# # @always_inline("nodebug")
# # def _accelerator_arch() -> StringLiteral:
# #     return __mlir_attr.`#kgen.param.expr<accelerator_arch> : !kgen.string`

# # def _get_arch[target: __mlir_type.`!kgen.target`]() -> StringLiteral:
# #     return __mlir_attr[
# #         `#kgen.param.expr<target_get_field,`,
# #         target,
# #         `, "arch" : !kgen.string`,
# #         `> : !kgen.string`,
# #     ]
# # from gpu.host._compile import _get_gpu_target, _get_info_from_target


# # def main():
# #    # alias code = compile._internal_compile_code[a]()
# #     #print(compile._internal_compile_code[a, emission_kind="llvm"]())
# #     # alias a = (compile._internal_compile_code[__fma[DType.float32], emission_kind="asm"]())
# #     # with open("./fma.asm", "w") as file:
# #     #     file.write(a)
# #     print(compile_info[__fma[DType.float32], target = _get_gpu_target()]().function_name)
# #     print(get_linkage_name[__fma[DType.float32]]())
# # # from gpu.host._compile import _ptxas_compile
# from memory import UnsafePointer
# from sys.ffi import external_call, _mlirtype_is_eq
# from sys.info import is_gpu, alignof, is_nvidia_gpu
# from _mlir import Type
# from gpu.host._compile import _get_gpu_target
# from _mlir._c.BuiltinTypes import MlirContext
# alias _must_be_mut_err = "UnsafePointer must be mutable for this operation"

# @parameter
# @always_inline
# def getstaticvalue[i: Int]() -> __mlir_type.i8:
#     return __mlir_op.`index.casts`[_type = __mlir_type.i8](i.__mlir_index__())

# @always_inline
# @parameter
# def toi8(i: UInt8) -> __mlir_type.ui8:
#     return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.ui8](i.value)

# @value
# @register_passable("trivial")
# struct LLvmPointer[mut: Bool = True, origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result](Stringable, Writable):
#     alias __ptr_type = __mlir_type.`!llvm.ptr`

#     # ===-------------------------------------------------------------------===#
#     # Fields
#     # ===-------------------------------------------------------------------===#

#     var __address: Self.__ptr_type
#     """The underlying pointer."""

#     # ===-------------------------------------------------------------------===#
#     # Life cycle methods
#     # ===-------------------------------------------------------------------===#

#     @always_inline
#     def __init__(out self):
#         """Create a null pointer."""
#         self.__address = __mlir_op.`llvm.mlir.zero`[_type = Self.__ptr_type]()

#     @doc_private
#     @always_inline
#     @implicit
#     def __init__(out self, value: Self.__ptr_type):
#         """Create a pointer with the input value.

#         Args:
#             value: The MLIR value of the pointer to construct with.
#         """
#         self.__address = value

#     @always_inline
#     def __init__(out self, *, other: Self):
#         """Copy the object.

#         Args:
#             other: The value to copy.
#         """
#         self.__address = other.__address

#     # ===-------------------------------------------------------------------===#
#     # Factory methods
#     # ===-------------------------------------------------------------------===#

#     @staticmethod
#     @always_inline
#     def alloc[alignment: Int = 1](size: Int32 = 1) -> Self:
#         """
#         Allocates static memory (stack) with the given alignment and size.
#         """
#         var __size = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](size.value)
#         alias __alignment = getstaticvalue[alignment]()
#         var ptr: Self.__ptr_type = __mlir_op.`llvm.alloca`[
#             _type = Self.__ptr_type,
#             _alignment=__alignment,
#             elem_type = __mlir_attr[__mlir_type.`ui8`],
#         ](__size)
#         var new = Self(ptr)
#         return new

#     @staticmethod
#     @always_inline
#     def malloc[alignment: Int = 1](size: Int32) -> Self:
#         """
#         Allocates dynamic memory (heap) with the given alignment and size.
#         """
#         var out = Self()
#         out.__address = external_call["malloc", Self.__ptr_type](size)
#         return out

#     @no_inline
#     def __str__(self) -> String:
#         return hex(int(self))

#     @no_inline
#     def write_to[W: Writer](self, mut writer: W):
#         writer.write(str(self))

#     @always_inline
#     def __int__(self) -> Int:
#         return __mlir_op.`index.casts`[_type = __mlir_type.index](
#             __mlir_op.`llvm.ptrtoint`[_type = __mlir_type.i64](self.__address)
#         )

#     @always_inline("nodebug")
#     def __eq__(self, rhs: Self) -> Bool:
#         """Returns True if the two pointers are equal.

#         Args:
#             rhs: The value of the other pointer.

#         Returns:
#             True if the two pointers are equal and False otherwise.
#         """
#         return int(self) == int(rhs)

#     @always_inline("nodebug")
#     def __ne__(self, rhs: Self) -> Bool:
#         """Returns True if the two pointers are not equal.

#         Args:
#             rhs: The value of the other pointer.

#         Returns:
#             True if the two pointers are not equal and False otherwise.
#         """
#         return int(self) != int(rhs)

#     # @always_inline("nodebug")
#     # def offset(self, index: Int) -> Self:
#     #     return __mlir_op.`llvm.getelementptr`[_type = Self.__ptr_type, elem_type = __mlir_attr[__mlir_type.`!llvm.ptr`], rawConstantIndices=_toi32(3)](self.__address)


# def _fromi32(val: __mlir_type.i32) -> Int:
#     return  __mlir_op.`index.casts`[_type = __mlir_type.index](val)

# def addveckernel(read A: UnsafePointer[Float32, mut = False], read B: UnsafePointer[Float32, mut = False], mut C: UnsafePointer[Float32], read n:Int):
#     @parameter
#     if is_nvidia_gpu():
#         var tid = ThreadIdx.x()
#         var bid = BlockIdx.x()
#         var block_size = BlockDim.x()

#         var global_idx = tid + bid * block_size

#         if global_idx < n:
#             var a_val = A[global_idx]
#             var b_val = B[global_idx]
#             var c_val = a_val + b_val

#             C[global_idx] = c_val
#         return
#     return

# @parameter
# @always_inline("nodebug")
# def mlirconst[int: Int32 = 1]() -> __mlir_type.i64:
#     alias val = __mlir_attr[`3`]
#     return __mlir_op.`llvm.mlir.constant`[_type=__mlir_type.i64, value=val]()

# def getRuntimeMlirContext() -> MlirContext:
#     var context: MlirContext
#     __mlir_op.`lit.ownership.mark_initialized`(
#         __get_mvalue_as_litref(context)
#     )
#     context.ptr = external_call["KGEN_CompilerRT_AsyncRT_GetCurrentRuntime", UnsafePointer[NoneType]]()
#     return context

# def allocation() -> Int as c:
#     _a = UnsafePointer[Scalar[DType.uint8]].alloc(5)
#     _b = _a.__int__()
#     a = LLvmPointer.alloc(2)
#     b = a.__int__()
#     c = _b + b
#     c = a == a


# # def main():
# #     var a = LLvmPointer.alloc(8)
# #     print(str(a))
# #     print(
# #         compile._internal_compile_code[
# #            addveckernel, emission_kind="llvm", target=_get_gpu_target()
# #         ]()
# #     )

# def main():
#     alias runtime = _get_current_runtime()
#     print(runtime)

# @parameter
# @always_inline
# def getstaticvaluei32[i: Int]() -> __mlir_type.i32:
#     return __mlir_op.`index.casts`[_type = __mlir_type.i32](i.__mlir_index__())

# def _toi32(val : Int32) -> __mlir_type.i32:
#     return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i32](val.value)

# from memory import UnsafePointer
# from _mlir.builtin_types import DialectType, FunctionType, MLIR_func, Context, Type
# from _mlir.builtin_attributes import Attribute, BoolAttr, TypeAttr, StringAttr, DialectAttribute, BuiltinAttributes, BuiltinTypes
# from _mlir.diagnostics import DiagnosticSeverity, MlirLogicalResult
# from _mlir.ir import NamedAttribute, _WriteState, Dialect, DialectHandle, DialectRegistry, Location, Operation, _OpBuilderList, IR, Region, Value, Block, Module
# from _mlir.rewrite import Rewriter
# from sys.ffi import _Global, _mlirtype_is_eq, external_call
# from sys._io import stdout
# from sys._libc import exit
# from buffer import NDBuffer
# from utils import IndexList
# from register import register_internal

# from max.graph._c import MOF_LIB, dtype_new

# fn test_mlir():
#     # Create an MLIR context first
#     var context_ptr = BuiltinTypes.mlirContextCreate()

#     # Create some basic types
#     var i32_type = Type(BuiltinTypes.mlirIntegerTypeGet(context_ptr, 32))
#     var f32_type = Type(BuiltinTypes.mlirF32TypeGet(context_ptr))

#     # print("Created i32 type:", i32_type)
#     # print("Created f32 type:", f32_type)

#     var vector_type = create_vector_type(context_ptr, i32_type, 4)
#     # print("Created vector type", vector_type)
#     var rank_tens = create_tensor_type(context_ptr, i32_type, List[Int64](4, 4))
#     # print("Created tensor type", rank_tens)
#     # Clean up
#     _ = context_ptr.ptr + 1 - 1

#     BuiltinTypes.mlirContextDestroy(context_ptr)

# fn create_vector_type(context_ptr: BuiltinTypes.MlirContext, element_type: Type, size: Int) -> Type:
#     var shape = List[Int64]()
#     shape.append(size)
#     var vector_type = BuiltinTypes.mlirVectorTypeGet(
#         shape.__len__(),
#         shape.data,
#         element_type.c
#     )
#     return Type(vector_type)

# fn create_tensor_type(context_ptr: BuiltinTypes.MlirContext, element_type: Type, dims: List[Int64]) -> Type:
#     var tensor_type = BuiltinTypes.mlirRankedTensorTypeGet(
#         dims.__len__(),
#         dims.data,
#         element_type.c,
#         BuiltinAttributes.mlirAttributeGetNull()
#     )
#     return Type(tensor_type)


# @register_internal("builtin.get_buffer_data")
# @always_inline
# fn get_buffer_data(
#     buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin]
# ) -> UnsafePointer[UInt8]:
#     return buffer.data

# from sys.info import alignof

# fn main():

# from max.graph.type import Type
# from max.graph._c import dtype_new, MOF_LIB
# from max.graph.graph import Module

# from sys import size_of
# from collections import OptionalReg
# from utils import StaticTuple
# from mc10.utils.unique_ptr import UniquePointer, DeleterFnType
# from mc10.hardware.device import Device

# @register_passable("trivial")
# struct DataType[_comptime_dtype: OptionalReg[DType]]:
#     comptime is_comptime: Bool = Self._comptime_dtype is not None
#     alias _size = 0 if Self.is_comptime else 1
#     var _runtime_value: StaticTuple[DType, Self._size]

#     fn __init__(out self):
#         __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

#     @implicit
#     fn __init__(
#         out self: DataType[_comptime_dtype = OptionalReg[DType](None)],
#         var value: DType,
#     ):
#         self._runtime_value = type_of(self._runtime_value)(value)

#     @staticmethod
#     fn dtype() -> DType:
#         constrained[Self.is_comptime, "dtype must be a comptime value"]()
#         return Self._comptime_dtype.value()

#     @always_inline
#     fn dtype(self) -> DType:
#         @parameter
#         if Self.is_comptime:
#             return Self._comptime_dtype.value()
#         else:
#             return self._runtime_value[0]


# fn test_dtype():
#     # alias type = DataType[DType.int].dtype()
#     # var typesi = DataType(DType.int)
#     # types = typesi.dtype()
#     # print(types)
#     # print(type)

#     # print("None: ", size_of[DataType[None]]())
#     # print("Typed: ", size_of[DataType[DType.int]]())
#     alias dt1 = DataType[DType.float32]()
#     print("Comptime dtype:", dt1.dtype())

#     var dt2 = DataType(DType.int64)
#     print("Runtime dtype:", dt2.dtype())


# @register_passable("trivial")
# struct Function[FnType: AnyTrivialRegType]:
#     var _fn: FnType
#     var _addr: UnsafePointer[Int, ImmutAnyOrigin]

#     fn __init__(out self, function: FnType):
#         self._fn = function

#         # Take the address of the function symbol itself
#         var fun: UnsafePointer[Int, ImmutAnyOrigin]
#         __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(fun))
#         UnsafePointer(to=fun).bitcast[FnType]()[] = function
#         self._addr = fun

#     fn compare(read self, other: Function[Self.FnType]) -> Bool:
#         return self._addr == other._addr

#     fn compare_exchange(
#         mut self,
#         expected: Function[Self.FnType],
#         new: Self.FnType,
#     ) -> Bool:
#         if self._addr == expected._addr:
#             self._fn = new
#             var fun: UnsafePointer[Int, ImmutAnyOrigin]
#             __mlir_op.`lit.ownership.mark_initialized`(
#                 __get_mvalue_as_litref(fun)
#             )
#             UnsafePointer(to=fun).bitcast[FnType]()[] = new
#             self._addr = fun
#             return True
#         return False

#     fn get(read self, /) -> Self.FnType:
#         return self._fn


# from testing import assert_equal, assert_true, assert_false, TestSuite


# fn make_int_ptr(value: Int) -> UnsafePointer[Int, MutOrigin.external]:
#     var p = alloc[Int](1)
#     p[] = value
#     return p


# fn test_unique_basic() raises:
#     var p = make_int_ptr(42)
#     var up = UniquePointer[Int](p)
#     assert_true(Bool(up), "UniquePointer should be truthy when owning")
#     assert_equal(up[].__int__(), 42)

#     # Destructor should free `p` when `up` goes out of scope.
#     # (Hard to assert directly, but we’ll test via custom deleter below.)


# fn counting_deleter(mut ptr: UnsafePointer[Int, MutOrigin.external]):
#     # For illustration, you can keep a global or static side-channel for checks.
#     # Here just free, but in a real test we’d increment a counter.
#     ptr.free()


# fn test_unique_release_and_reset() raises:
#     var p = make_int_ptr(10)
#     var up = UniquePointer[Int](p, DeleterFnType[Int](counting_deleter))

#     # release() should give us the raw pointer and prevent deletion on dtor
#     var raw = up.release()
#     assert_false(Bool(up))
#     assert_equal(raw[].__int__(), 10)
#     raw.free()  # manual free to avoid leak

#     # reset() should delete the old pointer and take ownership of the new one
#     var p2 = make_int_ptr(20)
#     up.reset(p2)
#     assert_true(Bool(up))
#     assert_equal(up[].__int__(), 20)
#     # when `up` goes out of scope, counting_deleter is called once for p2


# fn test_unique_move() raises:
#     var p = make_int_ptr(7)
#     var up1 = UniquePointer[Int](p)

#     # move-init up2 from up1
#     var up3 = up1^  # triggers __moveinit__
#     # assert_false(Bool(up2))  # moved-from should be null (conceptually)
#     assert_true(Bool(up3))
#     assert_equal(up3.get()[], 7)


# fn test_unique_pointer_move_constructor() raises:
#     """Test move constructor."""
#     var ptr1 = UniquePointer[Int](make_int_ptr(7))
#     var ptr2 = ptr1^
#     assert_true(ptr2)


from mc10.hardware.device import Device


struct Function[FnType: AnyType & ImplicitlyCopyable & Movable](
    AnyType & ImplicitlyCopyable & Movable
):
    """
    A lightweight wrapper around a function pointer, allowing for safe storage and comparison of function addresses.

    Parameters:
        FnType: The type of the function being wrapped.

    A small suggestion:
    This struct can be particularly useful in scenarios where function pointers need to be stored, compared, or passed around safely, such as in callback registries or function dispatch tables.
    Make sure to mark the function as `escaping`...

    eg.

    """

    var _fn: Optional[Self.FnType]
    var _addr: UnsafePointer[Int, ImmutAnyOrigin]

    fn __init__(out self):
        """Create a null Function."""
        self._fn = None
        self._addr = UnsafePointer[Int, ImmutAnyOrigin]()

    @implicit
    fn __init__(out self, value: NoneType):
        """Create a null Function."""
        self._fn = None
        self._addr = UnsafePointer[Int, ImmutAnyOrigin]()

    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        self = Self(value=NoneType(value))

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

    fn isInitialized(read self) -> Bool:
        return self._fn is not None

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


trait Allocator(ImplicitlyCopyable):
    """Abstract allocator trait for memory allocation.

    Similar to PyTorch's Allocator interface, this trait defines the interface
    for memory allocation and deallocation on different devices.
    """

    @staticmethod
    fn test(t: String):
        print(t)

    @staticmethod
    fn allocate(
        size: UInt, alignment: Int
    ) -> OpaquePointer[MutOrigin.external]:
        """Allocate memory on the device.

        Args:
            size: Number of bytes to allocate.
            alignment: Alignment requirement for the allocation.

        Returns:
            An OpaquePointer to the allocated memory.
        """
        ...

    @staticmethod
    fn device() -> Device:
        """Get the device this allocator manages.

        Returns:
            The Device this allocator operates on.
        """
        ...

    @staticmethod
    fn free(mut ptr: OpaquePointer[MutOrigin.external]) -> None:
        """Deallocate memory.

        Args:
            ptr : The OpaquePointer to deallocate.
        """
        ...

    @staticmethod
    fn name() -> StaticString:
        return "UnknownAllocator"


struct DummyAllocator(Allocator):
    """A dummy allocator that does not allocate any memory."""

    fn __init__(out self):
        pass

    @staticmethod
    fn allocate(
        size: UInt, alignment: Int
    ) -> OpaquePointer[MutOrigin.external]:
        """Dummy allocate that always fails."""
        return OpaquePointer[MutOrigin.external]()

    @staticmethod
    fn device() -> Device:
        """Get the device (CPU)."""
        return Device.CPU

    @staticmethod
    fn free(mut ptr: OpaquePointer[MutOrigin.external]) -> None:
        """Dummy free that does nothing."""
        pass

    @staticmethod
    fn name() -> StaticString:
        return "DummyAllocator"


# fn test_function[T: Allocator](all: T):

#     var f1 = Function[fn () -> None](all.test)
#     var f2 = Function[fn () capturing -> None](materialize[anylocal]())

#     print("Function f1 address:", (Int(UnsafePointer(to=f1.get()))))
#     print("Function f2 address:", (Int(UnsafePointer(to=f2.get()))))

#     var is_equal = f1.compare(f2)
#     print("Functions equal:", is_equal)

#     var exchanged = f1.compare_exchange(f2, materialize[anylocal]())
#     print("Exchange successful:", exchanged)

#     f1.get()()
#     f2.get()()


fn tryi() escaping -> None:
    print("helpp")


fn main():
    var fun1 = UnsafePointer(to=tryi)
    var fun2 = UnsafePointer(to=tryi)
    print(fun1 == fun2)
    print(fun1)
    print(fun2)
    fun1[]()
    fun2[]()


