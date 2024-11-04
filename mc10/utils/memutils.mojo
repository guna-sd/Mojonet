@always_inline
fn __malloc[Type: AnyTrivialRegType = NoneType](size: Int) -> UnsafePointer[Type]:
    return external_call[
            "malloc", UnsafePointer[NoneType]
        ](size).bitcast[Type]()

@always_inline
fn __calloc[Type: AnyTrivialRegType = NoneType](count: Int, size: Int = sizeof[Type]()) -> UnsafePointer[Type]:
    return external_call[
            "calloc", UnsafePointer[NoneType]
        ](count, size).bitcast[Type]()

@always_inline
fn __free[Type: AnyType](ptr: UnsafePointer[Type]):
    external_call["free", NoneType](ptr.bitcast[NoneType]())

@always_inline
fn __sizeof(type: DType) -> Int:
    return type.sizeof()