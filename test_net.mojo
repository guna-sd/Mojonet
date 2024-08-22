from net.tensor import Tensor, shape
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
        _ = (tensor1.bmm(tensor2))
    var end_time = time.now()
    print((end_time - start_time) / 1e9)

fn main() raises:
    bench()



def test_serialize():
    var shape = shape(4, 4, 4)
    var tensor = Tensor[DType.float32](shape).random()

    var serialized = Serialize.fromtensor(tensor)
    alias path = "./tensor_data.bin"
    serialized.write(path)

    var deserialized = Serialize.read(path)
    var deserialized_tensor = deserialized.totensor[DType.float32]()

    if tensor.__eq__(deserialized_tensor):
        print("Test passed: The original and deserialized tensors are equal.")
    else:
        print(
            "Test failed: The original and deserialized tensors are not equal."
        )


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
