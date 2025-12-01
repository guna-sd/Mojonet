# This falls under Base implementations but required a lot of study...
# For how to implement this safely...

from memory import UnsafePointer


struct _threadInternal:
    var _internal: UnsafePointer[Int32, ImmutAnyOrigin]


@explicit_destroy
struct Thread:
    var thread: _threadInternal
