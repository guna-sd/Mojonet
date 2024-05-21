from os.path import exists

alias SEEK_SET = 0
alias SEEK_END = 2

fn mkdir( path: String) -> Bool:
    """
    Create a directory at the given path.
    """
    if not exists(path):
        if external_call["mkdir", Int, DTypePointer[DType.int8]](path._as_ptr()) == 0:
            return True
        return False
    else:
        print("Directory already exists")
        return False

fn rmdir(path : String) -> Bool:
    """
    Remove a directory from the given path.
    """
    if exists(path):
        if external_call["rmdir", Int, DTypePointer[DType.int8]](path._as_ptr()) == 0:
            return True
        else:
            print("Directory is not empty")
            return False
    else:
        print("Path does not exist")
        return False

fn remove(path : String) -> Bool:
    """
    Remove a file from the given path.
    """
    if exists(path):
        if external_call["rm", Int, DTypePointer[DType.int8]](path._as_ptr()) == 0:
            return True
        else:
            print("Error: Cannot remove file " + path)
            return False
    else:
        print("Path does not exist")
        return False

struct FILE:
    ...

struct File:
    var fd : Pointer[FILE]

    fn __init__(inout self, path : String, mode : String):
        self.fd = external_call['fopen',Pointer[FILE], DTypePointer[DType.int8], DTypePointer[DType.int8]](path._as_ptr(),mode._as_ptr())
    
    fn read(self, size : Int = -1) -> String:
        var ssize = size
        if ssize == -1:
            ssize = self.size()
        var buffer = DTypePointer[DType.int8]().alloc(ssize)
        var ret = external_call['fread',Int32,DTypePointer[DType.int8],Int32,Int32,Pointer[FILE]](buffer,ssize,ssize,self.fd)
        if ret == -1:
            print("read failed")
        return String(buffer, int(ssize))

    fn size(self) -> Int:
        var result = external_call['fseek', Int, Pointer[FILE], Int, Int32](self.fd, 0, SEEK_END)
        if result != 0:
            print("Error seeking to end")
            return -1
        var size = external_call['ftell', Int, Pointer[FILE]](self.fd)
        if size == -1:
            print("Error getting file size")
            return -1
        result = external_call['fseek', Int, Pointer[FILE], Int, Int32](self.fd, 0, SEEK_SET)
        if result != 0:
            print("Error resetting to start")
            return -1

        return int(size)
    
    fn write(self, buffer : String):
        var ret = external_call['fwrite',Int32,DTypePointer[DType.uint8],Int32,Int32](buffer._as_ptr().bitcast[DType.uint8](),1, len(buffer),self.fd)
        if ret == -1:
            print("write failed")

    fn close(inout self):
        var ret = external_call['fclose',Int8,Pointer[FILE]](self.fd)
        if ret == -1:
            print("Failed to close")
    
    fn __copyinit__(inout self, new : Self):
        self.fd = new.fd
    
    fn __moveinit__(inout self, owned new : Self):
        self.fd = new.fd

    fn __enter__(owned self) -> Self:
        return self ^

fn fopen(path : String, mode : String) -> File:
    return File(path, mode)

struct Bytes:
    var data : List[UInt8]
  
    fn __init__(inout self):
        self.data = List[UInt8]()
  
    fn __init__(inout self, capacity : Int):
        self.data = List[UInt8](capacity=capacity)
  
    fn __init__(inout self, string: String):
        self.data = List[UInt8](capacity=len(string))
        for i in range(len(string)):
            self.data.append(ord(string[i]))
   
    fn __len__(self) -> Int:
        return self.data.__len__()

    fn append(inout self, value: UInt8):
        self.data.append(value)
    
    fn __setitem__(inout self, index: Int, value: UInt8):
        self.data[index] = value

    fn __getitem__(self, index: Int) -> UInt8:
        return self.data[index]
    
    fn __eq__(self, other: Self) -> Bool:
        if self.__len__() == other.__len__():
            for i in range(self.__len__()):
                if self[i] != other[i]:
                    return False
            return True
        return False

    fn __ne__(self, other: Self) -> Bool:
        if self.__eq__(other):
            return False
        return True

    fn __copyinit__(inout self, other: Self):
        self.data = other.data

    fn __moveinit__(inout self, owned other: Self):
        self.data = other.data

    fn __str__(self) -> String:
        var result: String = ""

        for i in range(self.__len__()):
            var val = self[i]
            if val != 0:
                result += chr(int(val))

        return result
    
    fn _ptr(self) ->DTypePointer[DType.uint8]:
        return rebind[DTypePointer[DType.uint8]](self.data.data)

fn tobytes[dtype: DType](value: Scalar[dtype]) -> Bytes:
    var bits = bitcast[DType.uint64](value.cast[type64[dtype]()]())
    var data = Bytes(capacity=8)
    for i in range(8):
        data.append(((bits >> (i * 8))).cast[DType.uint8]())
    return data

fn frombytes[dtype: DType](data: Bytes) -> Scalar[dtype]:
    if data.__len__() != 8:
        print("Invalid byte length for conversion")
        exit(1)

    var bits: UInt64 = 0
    for i in range(8):
        bits |= ((data[i]).cast[DType.uint64]()) << (i * 8)

    return bitcast[type64[dtype]()](bits).cast[dtype]()

fn type64[dtype : DType]() -> DType:
    @parameter
    if dtype.is_floating_point():
        return DType.float64
    elif dtype.is_signed():
        return DType.int64
    elif dtype.is_integral():
        return DType.uint64
    constrained[False, "Type must be numeric"]()
    return DType.invalid
