@always_inline
fn __malloc[Type: AnyType](size: Int) -> UnsafePointer[Type]:
    return external_call[
            "malloc", UnsafePointer[NoneType]
        ](size).bitcast[Type]()
@always_inline
fn __calloc[Type: AnyType](size: Int) -> UnsafePointer[Type]:
    return external_call[
            "calloc", UnsafePointer[NoneType]
        ](size, sizeof[Type]()).bitcast[Type]()

@always_inline
fn __free[Type: AnyType](ptr: UnsafePointer[Type]):
    external_call["free", NoneType](ptr.bitcast[NoneType]())

@always_inline
fn __sizeof(type: DType) -> Int:
    if type == DType.float16:
        return sizeof[DType.float16]()
    if type == DType.bfloat16:
        return sizeof[DType.bfloat16]()
    if type == DType.float32:
        return sizeof[DType.float32]()
    if type == DType.tensor_float32:
        return sizeof[DType.tensor_float32]()
    if type == DType.float64:
        return sizeof[DType.float64]()
    if type == DType.float8e4m3:
        return sizeof[DType.float8e4m3]()
    if type == DType.float8e5m2:
        return sizeof[DType.float8e5m2]()
    if type == DType.int8:
        return sizeof[DType.int8]()
    if type == DType.int16:
        return sizeof[DType.int16]()
    if type == DType.int32:
        return sizeof[DType.int32]()
    if type == DType.int64:
        return sizeof[DType.int64]()
    if type == DType.uint8:
        return sizeof[DType.uint8]()
    if type == DType.uint16:
        return sizeof[DType.uint16]()
    if type == DType.uint32:
        return sizeof[DType.uint32]()
    if type == DType.uint64:
        return sizeof[DType.uint64]()
    if type == DType.bool:
        return sizeof[DType.bool]()
    if type == DType.invalid:
        return 0
    else:
        return -1

