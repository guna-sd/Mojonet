from memory import UnsafePointer
from sys import (
    external_call,
    sizeof,
    llvm_intrinsic,
    is_gpu,
    alignof,
    PrefetchOptions,
    prefetch,
    PrefetchCache,
    PrefetchRW,
    PrefetchLocality,
)

alias kAlignment = 64

alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()


@always_inline
fn __malloc[
    Type: AnyTrivialRegType = NoneType
](size: Int) -> UnsafePointer[Type]:
    return external_call["malloc", UnsafePointer[NoneType]](size).bitcast[
        Type
    ]()


@always_inline
fn __calloc[
    Type: AnyTrivialRegType = NoneType
](count: Int, size: Int = sizeof[Type]()) -> UnsafePointer[Type]:
    return external_call["calloc", UnsafePointer[NoneType]](
        count, size
    ).bitcast[Type]()


@always_inline
fn __free[Type: AnyType](ptr: UnsafePointer[Type]):
    external_call["free", NoneType](ptr.bitcast[NoneType]())


@always_inline
fn __sizeof(type: DType) -> Int:
    return type.sizeof()


@always_inline
fn _default_alignment[type: AnyType]() -> Int:
    return alignof[type]() if is_gpu() else 1


@always_inline
fn _default_alignment[type: DType, width: Int = 1]() -> Int:
    return _default_alignment[Scalar[type]]()
