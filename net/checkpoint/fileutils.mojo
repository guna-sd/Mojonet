from os.path import exists
from os.pathlike import PathLike

alias SEEK_SET = 0
alias SEEK_END = 2
alias NBytes = DType.uint64.sizeof()


fn remove(path: String) -> Bool:
    if exists(path):
        if (
            external_call["unlink", Int, UnsafePointer[UInt8]](
                path.unsafe_ptr()
            )
            == 0
        ):
            return True
        else:
            return False
    else:
        print("Path does not exist")
        return False


fn remove[pathlike: PathLike](path: pathlike):
    var result = remove(path.__fspath__())
    if result:
        return
    else:
        print(result)


fn mkdir(path: String) -> Bool:
    """
    Create a directory at the given path.
    """
    if not exists(path):
        if (
            external_call["mkdir", Int, UnsafePointer[UInt8]](
                path.unsafe_ptr()
            )
            == 0
        ):
            return True
        return False
    else:
        print("Directory already exists")
        return False


fn mkdir[pathlike: PathLike](path: pathlike):
    var result = mkdir(path.__fspath__())
    if result:
        return
    else:
        print(result)


fn rmdir(path: String) -> Bool:
    """
    Remove a directory from the given path.
    """
    if exists(path):
        if (
            external_call["rmdir", Int, UnsafePointer[UInt8]](
                path.unsafe_ptr()
            )
            == 0
        ):
            return True
        else:
            print("Directory is not empty")
            return False
    else:
        print("Path does not exist")
        return False


fn rmdir[pathlike: PathLike](path: pathlike):
    var result = rmdir(path.__fspath__())
    if result:
        return
    else:
        print(result)


struct FILE:
    ...


struct File:
    var fd: UnsafePointer[FILE]

    fn __init__(inout self, path: String, mode: String):
        self.fd = external_call[
            "fopen",
            UnsafePointer[FILE],
            UnsafePointer[UInt8],
            UnsafePointer[UInt8],
        ](path.unsafe_ptr(), mode.unsafe_ptr())

    fn read(self, size: Int = -1) -> String:
        var ssize = size
        if ssize == -1:
            ssize = self.size()
        var buffer = UnsafePointer[UInt8]().alloc(ssize)
        var ret = external_call[
            "fread",
            Int32,
            UnsafePointer[UInt8],
            Int32,
            Int32,
            UnsafePointer[FILE],
        ](buffer, 1, ssize, self.fd)
        if ret == -1:
            print("read failed")
        return String(buffer, ssize)

    fn readbytes(self, size: Int = -1) -> Bytes:
        var ssize = size
        if ssize == -1:
            ssize = self.size()
        var buffer = UnsafePointer[UInt8]().alloc(ssize)
        var ret = external_call[
            "fread",
            Int32,
            UnsafePointer[UInt8],
            Int32,
            Int32,
            UnsafePointer[FILE],
        ](buffer, 1, ssize, self.fd)
        if ret == -1:
            print("read failed")
        return Bytes(buffer)

    fn size(self) -> Int:
        var result = external_call[
            "fseek", Int, UnsafePointer[FILE], Int, Int32
        ](self.fd, 0, SEEK_END)
        if result != 0:
            print("Error seeking to end")
            return -1
        var size = external_call["ftell", Int, UnsafePointer[FILE]](self.fd)
        if size == -1:
            print("Error getting file size")
            return -1
        result = external_call["fseek", Int, UnsafePointer[FILE], Int, Int32](
            self.fd, 0, SEEK_SET
        )
        if result != 0:
            print("Error resetting to start")
            return -1
        return int(size + 1)

    fn write(self, buffer: String):
        var ret = external_call[
            "fwrite",
            Int32,
            UnsafePointer[UInt8],
            Int32,
            Int32,
            UnsafePointer[FILE],
        ](buffer.unsafe_ptr(), 1, len(buffer), self.fd)
        if ret == -1:
            print("write failed")

    fn writebytes(self, buffer: Bytes):
        var ret = external_call[
            "fwrite",
            Int32,
            UnsafePointer[UInt8],
            Int32,
            Int32,
            UnsafePointer[FILE],
        ](buffer.unsafe_ptr(), 1, len(buffer), self.fd)
        if ret == -1:
            print("write failed")

    fn close(inout self):
        var ret = external_call["fclose", Int32, UnsafePointer[FILE]](self.fd)
        if ret == -1:
            print("Failed to close")

    fn __copyinit__(inout self, new: Self):
        self.fd = new.fd

    fn __moveinit__(inout self, owned new: Self):
        self.fd = new.fd

    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self^


fn fopen(path: String, mode: String) -> File:
    return File(path, mode)


@register_passable("trivial")
struct Bytes(Sized, Stringable, Representable):
    var data: StaticTuple[UInt8, NBytes]

    @always_inline("nodebug")
    fn __init__(inout self):
        self.data = StaticTuple[UInt8, NBytes]()

    @always_inline("nodebug")
    fn __init__(inout self, bytes: UnsafePointer[UInt8]):
        var data = StaticTuple[UInt8, NBytes]()

        @parameter
        for i in range(NBytes):
            data[i] = bytes[i]
        self.data = data

    @always_inline("nodebug")
    fn __init__(inout self, owned bytes: List[UInt8]):
        var data = StaticTuple[UInt8, NBytes]()

        @parameter
        for i in range(NBytes):
            data[i] = bytes[i]
        self.data = data

    @always_inline("nodebug")
    fn __init__(inout self, string: String):
        var data = StaticTuple[UInt8, NBytes]()

        @parameter
        for i in range(NBytes):
            data[i] = string._buffer[i]
        self.data = data

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return len(self.data)

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: UInt8):
        self.data[index] = value

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> UInt8:
        return self.data[index]

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        if self.__len__() == other.__len__():
            for i in range(self.__len__()):
                if self[i] != other[i]:
                    return False
            return True
        return False

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        if self.__eq__(other):
            return False
        return True

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        var result: String = "["
        for i in range(self.__len__()):
            result += str(self[i])
            if i != self.__len__() - 1:
                result += ", "
        return result + "]"

    @always_inline("nodebug")
    fn __str__(self) -> String:
        var result: String = ""

        @parameter
        for i in range(NBytes):
            result += chr(int(self[i]))
        return result

    @always_inline("nodebug")
    fn clear(inout self):
        @parameter
        for i in range(NBytes):
            self[i] = 0

    @always_inline("nodebug")
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        var data = UnsafePointer[UInt8]().alloc(NBytes)

        @parameter
        for i in range(NBytes):
            data[i] = self[i]
        return data

    @always_inline("nodebug")
    fn list(self) -> List[UInt8]:
        var data = List[UInt8](capacity=NBytes)

        @parameter
        for i in range(NBytes):
            data[i] = self[i]
        return data

    @always_inline("nodebug")
    fn reverse(self) -> Self:
        var prev = self.list()
        prev.reverse()
        return Bytes(prev^)


fn tobytes[dtype: DType](value: Scalar[dtype]) -> Bytes:
    var bits = bitcast[DType.uint64](value.cast[type64[dtype]()]())
    var data = Bytes()

    @parameter
    for i in range(NBytes):
        data[i] = ((bits >> (i * 8))).cast[DType.uint8]()
    return data


fn frombytes[dtype: DType](data: Bytes) -> Scalar[dtype]:
    if not len(data) >= NBytes:
        print("Invalid byte length")
        exit(1)

    var bits = Scalar[DType.uint64]()

    @parameter
    for i in range(NBytes):
        bits |= ((data[i]).cast[DType.uint64]()) << (i * 8)

    return bitcast[type64[dtype]()](bits).cast[dtype]()


fn type64[dtype: DType]() -> DType:
    @parameter
    if dtype.is_floating_point():
        return DType.float64
    elif dtype.is_signed():
        return DType.int64
    elif dtype.is_integral():
        return DType.uint64
    constrained[False, "Type must be numeric"]()
    return DType.invalid
