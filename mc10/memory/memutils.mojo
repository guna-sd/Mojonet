from memory import UnsafePointer
from sys import (
    external_call,
    sizeof,
    llvm_intrinsic,
    simdbitwidth,
    is_gpu,
    alignof,
    prefetch,
    PrefetchOptions,
)
from math import (
    align_down,
    align_up,
)

alias kAlignment = 64
alias alignment: Int = 128 if is_gpu() else kAlignment

alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()


## TODO: This is a workaround for the fact, Not sure if this is the right implementation...
@always_inline
fn malloc(size: Int, /) -> __mlir_type[`!kgen.pointer<`, __mlir_type.ui8, `>`]:
    @parameter
    if is_gpu():
        return external_call[
            "aligned_alloc",
            __mlir_type[`!kgen.pointer<`, __mlir_type.ui8, `>`],
        ](alignment, size)
    else:
        return __mlir_op.`pop.aligned_alloc`[
            _type = __mlir_type[`!kgen.pointer<`, __mlir_type.ui8, `>`]
        ](alignment.value, size.value)


@always_inline
fn free(ptr: UnsafePointer):
    @parameter
    if is_gpu():
        external_call["free", NoneType](ptr.bitcast[NoneType]().address)
    else:
        __mlir_op.`pop.aligned_free`(ptr.address)
