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

fn mkdir( path: String, exists_ok : Bool) -> Bool:
    """
    Create a directory at the given path.
    """
    if not exists(path):
        if external_call["mkdir", Int, AnyPointer[Int8]](path._buffer.data) == 0:
            return True
        return False
    else:
        print("Directory already exists")
        return False

fn read_file(path : String) raises -> String:
    with File(path,'r') as file:
        return file.read()

fn write_file(content : String, path : String)raises:
    with File(path,'r') as file:
        file.write(content)

fn rmdir(path : String) -> Bool:
    """
    Remove a directory from the given path.
    """
    if exists(path):
        if external_call["rmdir", Int, AnyPointer[Int8]](path._buffer.data) == 0:
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
        if external_call["rm", Int, AnyPointer[Int8]](path._buffer.data) == 0:
            return True
        else:
            print("Error: Cannot remove file " + path)
            return False
    else:
        print("Path does not exist")
        return False

fn get_mode(mode : String) -> Int:
    if mode == "r":
        return RDONLY
    if mode == "w":
        return WRONLY
    if mode == "w+":
        return CREAT
    if mode == "r+":
        return RDWR
    if mode == "a":
        return APPEND
    return -1

struct File:
    var fd : Int32

    fn __init__(inout self, path : String, mode : String):
        var _mode = get_mode(mode)
        if _mode == -1:
            print("Invalid mode")
        self.fd = external_call['open',Int32, DTypePointer[DType.int8], Int8](path._as_ptr(),_mode)
        if self.fd == -1:
            print("Failed to open")
            abort(external_call["exit", Int](1))
    
    fn read(self, size : Int = -1) -> String:
        var ssize = size
        if ssize == -1:
            ssize = self.size()
        var buffer = DTypePointer[DType.int8]().alloc(ssize)
        var ret = external_call['read',Int32,Int32,DTypePointer[DType.int8],Int](self.fd,buffer,ssize)
        if ret == -1:
            print("read failed")
        return String(buffer, int(ssize))

    fn size(self) -> Int:
        var size = external_call['lseek',Int, Int32, Int8](self.fd,0,SEEK_END)
        var reset = external_call['lseek',Int,Int32, Int8](self.fd,0,SEEK_SET)
        return int(size)
    
    fn write(self, buffer : String):
        var ret = external_call['write',Int,Int32,DTypePointer[DType.int8],Int](self.fd,buffer._as_ptr(), len(buffer))
        if ret == -1:
            print("write failed")

    fn close(inout self):
        var ret = external_call['close',Int8,Int32](self.fd)
        if ret == -1:
            print("Failed to close")
    
    fn __copyinit__(inout self, new : Self):
        self.fd = new.fd
    
    fn __moveinit__(inout self, owned new : Self):
        self.fd = new.fd

    fn __enter__(owned self) -> Self:
        return self ^

fn main():
    with File("/home/guna/test.sh","w") as f:
        f.write("jenvbo")
