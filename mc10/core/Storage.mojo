from memory import UnsafePointer
from mc10.core.Shapes import dim, shape, index
from mc10.types.DataPointer import DataPointer, _default_alignment
from memory import stack_allocation
from utils import IndexList

@parameter
@always_inline("nodebug")
fn _compute_strides[rank: Int](shapes : shape) -> IndexList[rank]:
    var stride = 1
    var strides = IndexList[rank]()
    for i in range(rank - 1, -1, -1):
        strides[i] = stride
        stride *= shapes[i]
    return strides

@parameter
@always_inline("nodebug")
fn _compute_indices[rank: Int](shape: shape, index: Int) -> IndexList[rank]:
    var idx = index
    var indices = IndexList[rank]()
    for dim in reversed(shape.list()):
        indices[0] = (idx % dim[])
        idx //= dim[]
    return indices

@parameter
@always_inline("nodebug")
fn _flatten_index(shape: shape, *indices: Int) -> Int:
    var flat_index = 0
    var stride = 1
    for i in range(shape.rank() - 1, -1, -1):
        flat_index += indices[i] * stride
        stride *= shape[i]
    return flat_index

@value
@register_passable("trivial")
struct Buffer[
    type: DType,
    /,
    staticSize: dim = dim(),
    *,
    address_space: AddressSpace = AddressSpace(0),
    origin: MutableOrigin = MutableAnyOrigin,
]:
    var data: DataPointer[address_space=address_space, origin=origin]
    var dynamicSize: index
    var dtype: DType

    fn __init__(out self):
        self.data = DataPointer[address_space=address_space, origin=origin]()
        self.dynamicSize = 0
        self.dtype = DType.invalid

    @implicit
    fn __init__(
        out self,
        ref ptr: UnsafePointer[SIMD[type, 1], address_space=address_space],
    ):
        self.data = ptr
        self.dtype = type
        self.dynamicSize = Int(staticSize)

    fn __init__(
        out self,
        ptr: UnsafePointer[SIMD[type, 1], address_space=address_space],
        size: Int,
    ):
        self.data = ptr
        self.dtype = type
        self.dynamicSize = size

    fn __init__(
        out self, ptr: DataPointer[address_space=address_space], dtype: DType
    ):
        self.data = ptr
        self.dtype = dtype
        self.dynamicSize = Int(staticSize)

    fn __init__(
        out self,
        ptr: DataPointer[address_space=address_space],
        dtype: DType,
        size: Int,
    ):
        self.data = ptr
        self.dtype = dtype
        self.dynamicSize = size

    fn __getitem__(self, idx: index) -> Scalar[type]:
        return self.data.__getitem__[type](Int(idx))

    fn __setitem__(self, idx: index, val: Scalar[type]):
        self.data.__getitem__[type](Int(idx)) = val

    fn zero(self):
        @parameter
        if not staticSize.is_dynamic():

            @parameter
            for i in range(staticSize):
                self[i] = 0
            return

        for i in range(self.dynamicSize):
            self[i] = 0

    fn fill(self, val: Scalar[type]):
        @parameter
        if not staticSize.is_dynamic():

            @parameter
            for i in range(staticSize):
                self[i] = val
            return

        for i in range(self.dynamicSize):
            self[i] = val

    @staticmethod
    fn staticAlloc[alignment: Int = _default_alignment[type]()]() -> Self:
        return Self(
            stack_allocation[
                staticSize,
                type,
                alignment=alignment,
                address_space=address_space,
            ]()
        )


@value
@register_passable("trivial")
struct NDBuffer[
    type: DType,
    rank: Int,
    /,
    staticShape: shape = shape.unknown[rank](),
    staticStrides: shape = shape.unknown[rank](),
    *,
    alignment: Int = 1,
    address_space: AddressSpace = AddressSpace(0),
    exclusive: Bool = True,
]:
    var dynamicShape: IndexList[rank, unsigned=True]
    var dynamicStrides: IndexList[rank, unsigned=True]
    var data: UnsafePointer[
        SIMD[type, 1], address_space=address_space, alignment=alignment
    ]