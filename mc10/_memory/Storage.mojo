## Note: Not sure if this is the right place for this file, but it is a good start
## TODO: Complete the rest of the Implementation for the Buffer and NDBuffer and add the rest of the utility methods

from memory import UnsafePointer
from mc10.types.Shapes import dim
from utils import IndexList


# A temporary alias later make a decision whether to have a custom type or IndexList as a default ??
comptime shape = List[Int]


@parameter
@always_inline("nodebug")
fn _compute_strides[rank: Int](shapes: shape) -> IndexList[rank]:
    var stride = 1
    var strides = IndexList[rank]()
    for i in range(rank - 1, -1, -1):
        strides[i] = stride
        stride *= Int(shapes[i])
    return strides


@parameter
@always_inline("nodebug")
fn _compute_indices[rank: Int](shape: shape, index: Int) -> IndexList[rank]:
    var idx = index
    var indices = IndexList[rank]()
    for dim in reversed(shape):
        indices[0] = idx % Int(dim)
        idx //= Int(dim)
    return indices


@parameter
@always_inline("nodebug")
fn _flatten_index(shape: shape, *indices: Int) -> Int:
    var flat_index = 0
    var stride = 1
    for i in range(shape.__len__() - 1, -1, -1):
        flat_index += indices[i] * stride
        stride *= Int(shape[i])
    return flat_index


## TODO: This is still a workaround for the fact

# @value
# @register_passable("trivial")
# struct NDBuffer[
#     type: DType,
#     rank: Int,
#     /,
#     staticShape: shape = shape.unknown[rank](),
#     staticStrides: shape = shape.unknown[rank](),
#     *,
#     alignment: Int = 1,
#     address_space: AddressSpace = AddressSpace(0),
#     exclusive: Bool = True,
# ]:
#     var dynamicShape: IndexList[rank, element_type = DType.uint64]
#     var dynamicStrides: IndexList[rank, element_type = DType.uint64]
#     var data: UnsafePointer[
#         SIMD[type, 1], address_space=address_space
#     ]


# @value
# @register_passable("trivial")
# struct Buffer[dtype: DType, rank: Int]:
#     alias dyn = IndexList[rank, element_type = DType.uint64]

#     var data: UnsafePointer[Scalar[dtype]]
#     """The underlying data for the buffer. The pointer is not owned by the
#     NDBuffer."""
#     var dynamic_shape: IndexList[rank, element_type = DType.uint64]
#     """The dynamic value of the shape."""
#     var dynamic_stride: IndexList[rank, element_type = DType.uint64]
#     """The dynamic stride of the buffer."""

#     fn __init__(out self):
#         self.data = UnsafePointer[Scalar[dtype]]()
#         self.dynamic_shape = Self.dyn()
#         self.dynamic_stride = Self.dyn()


# struct StorageImpl:
#     var ptr: UnsafePointer[Byte]
#     var size: Int
#     var device_id: Int
#     # Critical: How do we free 'ptr' when refcount hits 0?
#     # Option A: Store a pointer to the allocator here (bloat).
#     # Option B: The Global Pool is a true Singleton (easier, scale risks).

#     fn __del__(deinit self):
#         # Logic to return memory to the pool
#         global_pool_free(self.ptr, self.device_id)
