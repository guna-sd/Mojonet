from net.tensor import Tensor, shape, MojoTensor
from net.checkpoint import *


fn main() raises:
    ...


def test_mkdir():
    path = "test_dir"
    if mkdir(path) == True:
        print("mkdir ok")
    if exists(path) == True:
        print("exists ok")
    if mkdir(path) == False:
        print("mkdir fail ok")
    print(rmdir(path))

def test_remove():
    path = "test_file.txt"
    with fopen(path, "w") as f:
        f.write(str("test"))
    remove(path)

def test_file_read():
    path = "test_file.txt"
    with open(path, "w") as f:
        f.write(str("test"))
    file = fopen(path, "r")
    if file.read() == "test":
        print("file read ok")
    remove(path)

def test_file_write():
    path = "test_file.txt"
    file = fopen(path, "w")
    file.write("test")
    file.close()
    with open(path, "r") as f:
        if f.read() == "test":
            print("file read ok")
    remove(path)