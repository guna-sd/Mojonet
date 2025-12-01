from builtin._location import __call_location, _SourceLocation
from pathlib import Path
from sys import external_call
from io.write import _WriteBufferHeap
from sys._build import is_debug_build
from io.io import _printf


@no_inline
fn __abort[
    origin: ImmutOrigin, *Ts: Writable
](messages: VariadicPack[_, origin, Writable, *Ts], loc: _SourceLocation):
    var buffer = _WriteBufferHeap()
    buffer.write("At ", loc, ": \n\t\t\t")
    buffer.write("\033[0;31mAssert Error: ")

    @parameter
    for i in range(messages.__len__()):
        messages[i].write_to(buffer)
    buffer.write("\033[0m\n")
    _printf["At: %s:%llu:%llu: Assert Error: %s\n"](
        loc.file_name.unsafe_ptr(),
        loc.line,
        loc.col,
        buffer,
    )
    abort()


@always_inline
fn asserts[*Ts: Writable, cond: Bool](*messages: *Ts):
    @parameter
    if cond:
        return
    __abort(messages, __call_location())


@always_inline
fn asserts[*Ts: Writable](cond: Bool, *messages: *Ts):
    if cond:
        return
    __abort(messages, __call_location())


@always_inline
fn abort[*Ts: Writable](*messages: *Ts):
    __abort(messages, __call_location())


# TODO: work around this part later.... 
# @always_inline
# fn abort(stackTrace: Bool):
#     if stackTrace:
#         var buffer = UnsafePointer[UInt8]()
#         var num_bytes = external_call["KGEN_CompilerRT_GetStackTrace", Int](
#             UnsafePointer(to=buffer), 0
#         )

#         if num_bytes == 0:
#             return abort("")

#         return abort(String(unsafe_from_utf8_ptr=buffer))
