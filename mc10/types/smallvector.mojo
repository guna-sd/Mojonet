from memory import UnsafePointer
from mc10.utils.debuggable import asserts, abort
from sys.info import sizeof

alias kDimVectorStaticSize = 5

@parameter
@always_inline
fn _int_type_of_width[width: Int]() -> DType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return DType.int8
    elif width == 16:
        return DType.int16
    elif width == 32:
        return DType.int32
    else:
        return DType.int64


@parameter
@always_inline
fn _uint_type_of_width[width: Int]() -> DType:
    constrained[
        width == 8 or width == 16 or width == 32 or width == 64,
        "width must be either 8, 16, 32, or 64",
    ]()

    @parameter
    if width == 8:
        return DType.uint8
    elif width == 16:
        return DType.uint16
    elif width == 32:
        return DType.uint32
    else:
        return DType.uint64


struct SmallVector[_bit_width: Int = DType.int32.bitwidth()]:
    alias typeSize = sizeof[SIMD[_int_type_of_width[_bit_width](), 1]]()
    var data: UnsafePointer[NoneType]
    var size: SIMD[_int_type_of_width[_bit_width](), 1]
    var capacity: SIMD[_int_type_of_width[_bit_width](), 1]

    @staticmethod
    fn getNewCapacity(minSize: Int, oldCapacity: Int) -> Int:
        alias maxSize = Int(
            SIMD[_int_type_of_width[_bit_width](), 1].MAX_FINITE
        )

        asserts(
            minSize > maxSize,
            "SmallVector unable to grow. Requested capacity (",
            minSize,
            ") is larger than maximum value for size type (",
            maxSize,
            ")",
        )

        if oldCapacity == maxSize:
            abort(
                "SmallVector capacity unable to grow. Already at maximum size ",
                maxSize,
            )

        newCapacity = 2 * oldCapacity + 1
        return min(max(newCapacity, minSize), maxSize)

    fn maxSize(self) -> Int:
        return Int(SIMD[_int_type_of_width[_bit_width](), 1].MAX_FINITE)

    fn mallocForGrow(self, minSize: Int) -> UnsafePointer[NoneType]:
        newCapacity = Self.getNewCapacity(minSize, Int(self.capacity))
        ptr = UnsafePointer[NoneType].alloc(Self.typeSize * newCapacity)
        if ptr == UnsafePointer[NoneType]():
            abort("SmallVector unable to grow. Memory Allocation Failed!")
        return ptr