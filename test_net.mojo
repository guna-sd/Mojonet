from net.tensor import Tensor, shape, MojoTensor
from net.checkpoint import *
from net import bmm
import time

alias b = 5
alias m = 1024
alias n = 2048
alias p = 1024


fn bench():
    var A = Tensor[DType.float32](b,m,n).random()
    var B = Tensor[DType.float32](b,n,p).random()
    benchmark_matmul(A, B)


fn benchmark_matmul[
    T: DType,
](tensor1: Tensor[T], tensor2: Tensor[T], num_iterations: Int = 5):
    print("start")
    var start_time = time.now()
    for _ in range(num_iterations):
        _ = (tensor1.fusedbmm[net.gelu](tensor2))
    var end_time = time.now()
    print((end_time - start_time))

fn main() raises:
    bench()


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
