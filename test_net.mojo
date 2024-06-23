from net.tensor import Tensor, shape, MojoTensor
from net.checkpoint import *


fn main() raises:
    test_serialize()


def test_serialize():
    var shape = shape(4, 4, 4)
    var tensor = Tensor[DType.float32](shape).random()

    var serialized = Serialize.fromtensor(tensor)
    var path = "./tensor_data.bin"
    serialized.write(path)

    var deserialized = Serialize.read(path)
    var deserialized_tensor = deserialized.totensor[DType.float32]()

    if tensor.__eq__(deserialized_tensor):
        print("Test passed: The original and deserialized tensors are equal.")
    else:
        print(
            "Test failed: The original and deserialized tensors are not equal."
        )
    debug_assert(remove(path), "Not removed")


def test_mkdir():
    path = "test_dir"
    if mkdir(path) == True:
        print("mkdir ok")
    if exists(path) == True:
        print("exists ok")
    if mkdir(path) == False:
        print("mkdir fail ok")
    debug_assert(remove(path), "Not removed")


def test_remove():
    path = "test_file.txt"
    with fopen(path, "w") as f:
        f.write(str("test"))
    debug_assert(remove(path), "Not removed")


def test_file_read():
    path = "test_file.txt"
    with open(path, "w") as f:
        f.write(str("test"))
    file = fopen(path, "r")
    if file.read() == "test":
        print("file read ok")
    debug_assert(remove(path), "Not removed")


def test_file_write():
    path = "test_file.txt"
    file = fopen(path, "w")
    file.write("test")
    file.close()
    with open(path, "r") as f:
        if f.read() == "test":
            print("file read ok")
    debug_assert(remove(path), "Not removed")
