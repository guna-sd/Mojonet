# Has to study this part cause we actually dont need a seperate runtime from the main itself
# TODO: has to find a way to design this in such a way so that we can use JIT and other compilations... for cached execution

# from sys import external_call
# from sys.ffi import _get_global, _get_global_or_null, _Global
# from memory import UnsafePointer
# from os import abort


# fn _init_global_runtime() -> OpaquePointer[MutOrigin.external]:
#     return external_call[
#         "KGEN_CompilerRT_AsyncRT_CreateRuntime",
#         OpaquePointer[MutOrigin.external],
#     ](0)


# fn _destroy_global_runtime(ptr: OpaquePointer[MutOrigin.external]):
#     """Destroy the global runtime if ever used."""
#     external_call["KGEN_CompilerRT_AsyncRT_DestroyRuntime", NoneType](ptr)


# @always_inline
# fn _get_current_or_global_runtime() -> OpaquePointer[MutOrigin.external]:
#     var current_runtime = external_call[
#         "KGEN_CompilerRT_AsyncRT_GetCurrentRuntime",
#         OpaquePointer[MutOrigin.external],
#     ]()
#     if current_runtime:
#         return current_runtime
#     return _get_global[
#         "_NetRuntime", _init_global_runtime, _destroy_global_runtime
#     ]()




# @fieldwise_init
# @register_passable("trivial")
# struct ComputeRuntime(Movable):
#     comptime _global_runtime = _Global[
#         "mojonet.compute_runtime",
#         _init_global_runtime,
#     ]

#     var _handle: OpaquePointer[MutOrigin.external]

#     @staticmethod
#     fn runtime() -> ComputeRuntime:
#         return ComputeRuntime(Self.get_or_create())

#     @staticmethod
#     fn get_or_create() -> OpaquePointer[MutOrigin.external]:
#         return _get_global[
#             "mojonet.compute_runtime",
#             _init_global_runtime,
#             _destroy_global_runtime,
#         ]()


# @register_passable("trivial")
# struct RuntimeContext:
#     var runtimePtr: OpaquePointer[MutOrigin.external]

#     @always_inline
#     fn __init__(out self):
#         """Initialize with current or global runtime."""
#         self.runtimePtr = _get_current_or_global_runtime()
