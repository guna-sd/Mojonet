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

from memory import UnsafePointer
from _mlir.builtin_types import DialectType, FunctionType, MLIR_func, Context, Type
from _mlir.builtin_attributes import Attribute, BoolAttr, TypeAttr, StringAttr, DialectAttribute, BuiltinAttributes, BuiltinTypes
from _mlir.diagnostics import DiagnosticSeverity, MlirLogicalResult
from _mlir.ir import NamedAttribute, _WriteState, Dialect, DialectHandle, DialectRegistry, Location, Operation, _OpBuilderList, IR, Region, Value, Block, Module
from _mlir.rewrite import Rewriter
from sys.ffi import _Global, _mlirtype_is_eq, external_call
from sys._io import stdout
from sys._libc import exit
from buffer import NDBuffer
from utils import IndexList
from register import register_internal

from max.graph._c import MOF_LIB, dtype_new

fn test_mlir():
    # Create an MLIR context first
    var context_ptr = BuiltinTypes.mlirContextCreate()
    
    # Create some basic types
    var i32_type = Type(BuiltinTypes.mlirIntegerTypeGet(context_ptr, 32))
    var f32_type = Type(BuiltinTypes.mlirF32TypeGet(context_ptr))

    # print("Created i32 type:", i32_type)
    # print("Created f32 type:", f32_type)

    var vector_type = create_vector_type(context_ptr, i32_type, 4)
    # print("Created vector type", vector_type)
    var rank_tens = create_tensor_type(context_ptr, i32_type, List[Int64](4, 4))
    # print("Created tensor type", rank_tens)
    # Clean up
    _ = context_ptr.ptr + 1 - 1

    BuiltinTypes.mlirContextDestroy(context_ptr)

fn create_vector_type(context_ptr: BuiltinTypes.MlirContext, element_type: Type, size: Int) -> Type:
    var shape = List[Int64]()
    shape.append(size)
    var vector_type = BuiltinTypes.mlirVectorTypeGet(
        shape.__len__(), 
        shape.data, 
        element_type.c
    )
    return Type(vector_type)

fn create_tensor_type(context_ptr: BuiltinTypes.MlirContext, element_type: Type, dims: List[Int64]) -> Type:
    var tensor_type = BuiltinTypes.mlirRankedTensorTypeGet(
        dims.__len__(),
        dims.data,
        element_type.c,
        BuiltinAttributes.mlirAttributeGetNull()
    )
    return Type(tensor_type)



@register_internal("builtin.get_buffer_data")
@always_inline
fn get_buffer_data(
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin]
) -> UnsafePointer[UInt8]:
    return buffer.data

from sys.info import alignof

fn main():
    