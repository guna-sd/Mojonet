@always_inline
fn __alignment(type: DType) -> Int:
    
    if triple_is_nvidia_cuda():
        if type == DType.float16:
            return alignof[DType.float16]()
        if type == DType.bfloat16:
            return alignof[DType.bfloat16]()
        if type == DType.float32:
            return alignof[DType.float32]()
        if type == DType.tensor_float32:
            return alignof[DType.tensor_float32]()
        if type == DType.float64:
            return alignof[DType.float64]()
        if type == DType.float8e4m3:
            return alignof[DType.float8e4m3]()
        if type == DType.float8e5m2:
            return alignof[DType.float8e5m2]()
        if type == DType.int8:
            return alignof[DType.int8]()
        if type == DType.int16:
            return alignof[DType.int16]()
        if type == DType.int32:
            return alignof[DType.int32]()
        if type == DType.int64:
            return alignof[DType.int64]()
        if type == DType.uint8:
            return alignof[DType.uint8]()
        if type == DType.uint16:
            return alignof[DType.uint16]()
        if type == DType.uint32:
            return alignof[DType.uint32]()
        if type == DType.uint64:
            return alignof[DType.uint64]()
        if type == DType.bool:
            return alignof[DType.bool]() 
    else:
        return 1


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
        return sizeof[DType.bool]()  # Typically 1 byte for bool
    if type == DType.invalid:
        return 0
    else:
        return -1 # Return -1 for unrecognized or unknown types
