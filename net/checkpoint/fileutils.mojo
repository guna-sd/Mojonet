from os.path import exists
from pathlib import Path



alias RDONLY = 0
alias WRONLY = 1
alias RDWR = 2
alias APPEND = 8
alias CREAT = 512
alias SYNC = 8192
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

struct File:
    var fd : Pointer[Int]

    fn __init__(inout self, path : String, mode : String):
        self.fd = external_call['fopen',Pointer[Int], DTypePointer[DType.int8], DTypePointer[DType.int8]](path._as_ptr(),mode._as_ptr())
    
    fn read(self, size : Int = -1) -> String:
        var ssize = size
        if ssize == -1:
            ssize = self.size()
        var buffer = DTypePointer[DType.int8]().alloc(ssize)
        var ret = external_call['fread',Int32,DTypePointer[DType.int8],Int32,Int32,Pointer[Int]](buffer,ssize,ssize,self.fd)
        if ret == -1:
            print("read failed")
        return String(buffer, int(ssize))

    fn size(self) -> Int:
        var result = external_call['fseek', Int, Pointer[Int], Int, Int32](self.fd, 0, SEEK_END)
        if result != 0:
            print("Error seeking to end")
            return -1
        var size = external_call['ftell', Int, Pointer[Int]](self.fd)
        if size == -1:
            print("Error getting file size")
            return -1
        result = external_call['fseek', Int, Pointer[Int], Int, Int32](self.fd, 0, SEEK_SET)
        if result != 0:
            print("Error resetting to start")
            return -1

        return int(size)
    
    fn write(self, buffer : String):
        var ret = external_call['fwrite',Int32,DTypePointer[DType.uint8],Int32,Int32](buffer._as_ptr().bitcast[DType.uint8](),1, len(buffer),self.fd)
        if ret == -1:
            print("write failed")

    fn close(inout self):
        var ret = external_call['fclose',Int8,Pointer[Int]](self.fd)
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
