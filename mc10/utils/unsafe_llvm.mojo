from sys.ffi import external_call, _mlirtype_is_eq
from mc10.__mlir import _toi32, _toi8, vector1d, MlirType, i8, i32, i64
from sys.info import is_gpu, alignof
from memory import UnsafePointer
from mc10.utils.Int import int
from mc10.utils.debuggable import asserts

from memory.pointer import _GPUAddressSpace


@value
@register_passable("trivial")
struct LLvmPointer[
    address_space: AddressSpace = AddressSpace(1),
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    alias __ptr_type = __mlir_type.`!llvm.ptr<0>`

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self.__ptr_type
    """The underlying pointer."""

    var dtype: DType

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Create a null pointer."""
        self.address = __mlir_op.`llvm.mlir.zero`[_type = Self.__ptr_type]()
        self.dtype = DType.int8

    @always_inline
    fn __init__(out self, *, dtype: DType):
        """Create a null pointer."""
        self.address = __mlir_op.`llvm.mlir.zero`[_type = Self.__ptr_type]()
        self.dtype = dtype

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, address: Self.__ptr_type):
        """Create a pointer with address.

        Args:
            address: The MLIR value of the pointer to construct with.
        """
        self.address = address
        self.dtype = DType.int8

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, address: __mlir_type.`!llvm.ptr<1>`):
        """Create a pointer with address.

        Args:
            address: The MLIR value of the pointer to construct with.
        """
        self.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = Self.__ptr_type
        ](address)
        self.dtype = DType.int8

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, address: __mlir_type.`!llvm.ptr<3>`):
        """Create a pointer with address.

        Args:
            address: The MLIR value of the pointer to construct with.
        """
        self.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = Self.__ptr_type
        ](address)
        self.dtype = DType.int8

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, address: __mlir_type.`!llvm.ptr<4>`):
        """Create a pointer with address.

        Args:
            address: The MLIR value of the pointer to construct with.
        """
        self.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = Self.__ptr_type
        ](address)
        self.dtype = DType.int8

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, address: __mlir_type.`!llvm.ptr<5>`):
        """Create a pointer with address.

        Args:
            address: The MLIR value of the pointer to construct with.
        """
        self.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = Self.__ptr_type
        ](address)
        self.dtype = DType.int8

    @always_inline
    fn copy(self) -> Self:
        """Copy an existing pointer.

        Returns:
            A copy of the value.
        """
        return Self(self.address, self.dtype)

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn alloca[alignment: Int = 1, type: MlirType = i8](size: Int = 1) -> Self:
        """
        Allocates static memory (stack) with the given alignment and size.
        """
        var __size = _toi32(Int32(size))
        alias __alignment = _toi8(Int8(alignment))
        var ptr = __mlir_op.`llvm.alloca`[
            _type = __mlir_type.`!llvm.ptr<1>`,
            _alignment=__alignment,
            elem_type = __mlir_attr[type.Type],
        ](__size)
        var new = Self(ptr)
        return new

    @staticmethod
    @always_inline
    fn malloc[alignment: Int = 1](size: Int) -> Self:
        """
        Allocates dynamic memory (heap) with the given alignment and size.
        """
        var out = Self()
        out.address = external_call["malloc", Self.__ptr_type](size)
        return out

    @staticmethod
    fn fromUnsafePointer(ptr: UnsafePointer) -> Self:
        var llvmptr = Self()
        llvmptr.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = Self.__ptr_type
        ](ptr.address)
        return llvmptr

    @staticmethod
    fn toUnsafePointer[T: AnyType](self) -> UnsafePointer[T]:
        ptr = UnsafePointer[T]()
        ptr.address = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type = ptr._mlir_type
        ](self.address)
        return ptr

    # not working trying to figure out how to!!
    # fn gep(self, offset: Int) -> Self:
    #     return __mlir_op.`llvm.getelementptr`[
    #         _type = Self.__ptr_type,
    #         elem_type = __mlir_attr[__mlir_type.`i8`],
    #         rawConstantIndices = __mlir_attr[`array<i32:1>`],
    #     ](self.address, vector1d(offset))

    # ===-------------------------------------------------------------------===#
    # Operator dunders and Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __add__(self, offset: int) -> Self:
        return self.offset(offset)

    @always_inline
    fn __sub__(self, offset: int) -> Self:
        return self + (-1 * offset)

    @always_inline
    fn __iadd__(mut self, offset: int):
        self = self + offset

    @always_inline
    fn __isub__(mut self, offset: int):
        self = self - offset

    @always_inline
    fn __bool__(self) -> Bool:
        return (self.__int__()) != 0

    @always_inline
    fn __as_bool__(self) -> Bool:
        return self.__bool__()

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        return (self.__int__()) == (rhs.__int__())

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        return (self.__int__()) != (rhs.__int__())

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        return (self.__int__()) < (rhs.__int__())

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        return (self.__int__()) <= (rhs.__int__())

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        return (self.__int__()) > (rhs.__int__())

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        return (self.__int__()) >= (rhs.__int__())

    @always_inline
    fn __int__(self) -> int:
        return __mlir_op.`llvm.ptrtoint`[_type = __mlir_type.i32](self.address)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(hex(self.__int__().__int__()))

    @always_inline
    fn offset(self, idx: int) -> Self:
        var current = self.__int__()
        offsets = self.dtype.sizeof() * idx
        new = current + offsets
        return Self(
            __mlir_op.`llvm.inttoptr`[_type = Self.__ptr_type](new.value),
            self.dtype,
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    # not working trying to figure out how to!!
    @staticmethod
    fn castAddressspace[dest_addr: int](self) -> Self:
        return __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr<1>`](self.address)