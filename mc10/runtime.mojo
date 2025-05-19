from sys import external_call
from sys.ffi import OpaquePointer, _get_global, _get_global_or_null
from memory import UnsafePointer

fn _init_global_runtime(ignored: OpaquePointer) -> OpaquePointer:
    return external_call[
        "KGEN_CompilerRT_AsyncRT_CreateRuntime",
        OpaquePointer,
    ](0)


fn _destroy_global_runtime(ptr: OpaquePointer):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_AsyncRT_DestroyRuntime", NoneType](ptr)


@always_inline
fn _get_current_or_global_runtime() -> OpaquePointer:
    var current_runtime = external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime", OpaquePointer
    ]()
    if current_runtime:
        return current_runtime
    return _get_global[
        "_NetRuntime", _init_global_runtime, _destroy_global_runtime
    ]()


@register_passable("trivial")
struct RuntimeContext:
    var runtimePtr: OpaquePointer

    @always_inline
    fn __init__(out self):
        """Initialize with current or global runtime."""
        self.runtimePtr = _get_current_or_global_runtime()

