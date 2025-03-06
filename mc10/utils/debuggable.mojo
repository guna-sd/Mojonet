from builtin._location import __call_location, _SourceLocation
from builtin.debug_assert import *
from pathlib import Path
from sys._io import stdout
from sys._libc import exit

from utils.write import (
    _WriteBufferStack,
    write_args,
)


alias defined_mode = env_get_string["ASSERT", "safe"]()


@value
struct LogType:
    alias DEBUG = LogType(0)
    alias INFO = LogType(1)
    alias WARN = LogType(2)
    alias ERROR = LogType(3)
    var level: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.level == other.level

    fn __ne__(self, other: Self) -> Bool:
        return self.level != other.level


alias GLOBAL_LOG_LEVEL = LogType.DEBUG


fn _log(message: String, logType: LogType):
    if logType.level < GLOBAL_LOG_LEVEL.level:
        return

    var stdout = FileDescriptor(1)
    var buffer = _WriteBufferStack[4096](stdout)

    if logType == LogType.DEBUG:
        buffer.write("\033[1;94mDEBUG: ")
    elif logType == LogType.INFO:
        buffer.write("\033[1;97mINFO: ")
    elif logType == LogType.WARN:
        buffer.write("\033[1;93mWARN: ")
    elif logType == LogType.ERROR:
        buffer.write("\033[1;95mERROR: ")
    else:
        buffer.write("LOG: ")

    buffer.write(message + "\n")
    buffer.flush()


fn debug(message: String):
    _log(message, LogType.DEBUG)


fn info(message: String):
    _log(message, LogType.INFO)


fn warn(message: String):
    _log(message, LogType.WARN)


fn error(message: String):
    _log(message, LogType.ERROR)


@no_inline
fn __assert(messages: VariadicPack[_, Writable, *_], loc: _SourceLocation):
    var stdout = FileDescriptor(1)
    var buffer = _WriteBufferStack[4096](stdout)
    buffer.write("At ", loc, ": \n\t\t\t")

    @parameter
    if defined_mode == "warn":
        buffer.write("\033[0;37mAssert Warning: ")
    else:
        buffer.write("\033[0;31mAssert Error: ")
    write_args(buffer, messages, end="\033[0m\n")
    buffer.flush()

    @parameter
    if defined_mode != "warn":
        abort()

@no_inline
fn __abort(messages: VariadicPack[_, Writable, *_]):
    var stdout = FileDescriptor(1)
    var buffer = _WriteBufferStack[4096](stdout)
    write_args(buffer, messages, end="\n")
    buffer.flush()
    abort()


@always_inline
fn asserts[*Ts: Writable, cond: Bool](*messages: *Ts):
    @parameter
    if cond:
        return
    var loc: _SourceLocation = __call_location()
    __assert(messages, loc)


@always_inline
fn asserts[*Ts: Writable](cond: Bool, *messages: *Ts):
    if cond:
        return
    var loc: _SourceLocation = __call_location()
    __assert(messages, loc)

@always_inline
fn abort[*Ts: Writable](*messages: *Ts):
    __abort(messages)

# TODO: still under construction...
fn capture_backtrace():
    max_frames = 100
    buffer = UnsafePointer[UnsafePointer[UInt8]].alloc(max_frames)
    frame_count = backtrace(buffer, max_frames)
    if frame_count == 0:
        print("No stack frames captured.")
        buffer.free()
        return

    symbols = backtrace_symbols(buffer, frame_count)
    if not symbols:
        print("Failed to retrieve backtrace symbols.")
        buffer.free()
        return

    for i in range(frame_count):
        symbol = symbols[i]
        if symbol:
            print("Frame ", i, ": ", String(symbol))
        else:
            print("Frame ", i, ": [unknown]")

    buffer.free()


fn backtrace(buffer: UnsafePointer[UnsafePointer[UInt8]], size: Int) -> Int:
    return external_call[
        "backtrace", Int, UnsafePointer[UnsafePointer[UInt8]], Int
    ](buffer, size)


fn backtrace_symbols(
    buffer: UnsafePointer[UnsafePointer[UInt8]], size: Int
) -> UnsafePointer[UnsafePointer[Int8]]:
    return external_call[
        "backtrace_symbols",
        UnsafePointer[UnsafePointer[Int8]],
        UnsafePointer[UnsafePointer[UInt8]],
        Int,
    ](buffer, size)
