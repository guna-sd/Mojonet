from net.tensor import Tensor, shape, batch_matmul, matmul
from tensor import Tensor as _Tensor
import time
from net.checkpoint import *
from benchmark.benchmark import run
from python import Python

alias b = 3
alias m = 1024
alias n = 2048
alias p = 1024

var A = Tensor[DType.float32](b,m,n).random()
var B = Tensor[DType.float32](b,n,p).random()

fn main() raises:
    bench()

fn bench() raises:
    var benchmark = run[test_matrix_multipication]()
    var secs = benchmark.mean()
    benchmark.print_full()
    var gflops = ((3 * m * n * p) / secs) / 1e9

    var py = Python.import_module("builtins")
    _ = py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("bmm", gflops))

fn benchmark_matmul[T : DType, func : fn[dtype : DType](Tensor[dtype], Tensor[dtype]) -> Tensor[dtype]
    ](tensor1 : Tensor[T], tensor2 : Tensor[T], num_iterations : Int =10) -> Float64:

    var start_time = time.now()
    for _ in range(num_iterations):
        _ = func(tensor1, tensor2)
    var end_time = time.now()
    return (end_time - start_time) / 1e9

fn test_matrix_multipication():
    _ = batch_matmul(A,B)
    #print(benchmark_matmul[DType.float32, batch_matmul](A,B))
    

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