from tensor.tensor import Tensor, TensorShape, TensorSpec
from tensor import rand
from benchmark import benchmark 
from benchmark.compiler import keep
from random import seed

alias dtype = DType.float64
alias dim = 100
alias rounds = 100
alias mul = 0.99999234

struct S1:
    var data:Tensor[dtype]
    var type:String

    fn __init__(inout self,ts:TensorShape):
        self.data = Tensor[dtype](ts)
        self.type = "s1"
    
    fn __copyinit__(inout self, existing: Self):  
        self.data = existing.data  
        self.type = existing.type
       
    fn __mul__(self,x:Float64) -> Self:
        var out = S1(self.data.shape())
        out.data = self.data * x 
        return out

struct S2:
    var data : DTypePointer[dtype]
    var len:Int
    var type:String

    fn __init__(inout self,len:Int):
        self.data = DTypePointer[dtype].alloc(len)
        self.len = len
        self.type = "s2"

    fn __copyinit__(inout self, existing: Self):  
        self.data = existing.data  
        self.len = existing.len 
        self.type = existing.type

    fn __mul__(self,x:Float64) -> Self:
        var out = S2(self.len)
        out.data.store(self.data.load() * x) 
        return out

@register_passable("trivial")
struct S3:
    var data : DTypePointer[dtype]
    var len:Int
    var type:StringRef

    fn __init__(inout self,len:Int):
        self.data = DTypePointer[dtype].alloc(len)
        self.len = len
        self.type = "s3"

    fn __mul__(self,x:Float64) -> Self:
        var out = S3(self.len)
        out.data.store(self.data.load() * x) 
        return out

fn test() raises:
    seed(37)

    var ts = TensorShape(dim)
    var s1 = S1(ts)
    var s2 = S2(dim)
    var s3 = S3(dim)

    for i in range(dim):
        s1.data[i] = random.random_float64()
        s2.data[i] = s1.data[i]
        s3.data[i] = s1.data[i]

    @parameter
    fn _benchmark_s1():
        for i in range(rounds):
            s1 = s1 * mul
    @parameter
    fn _benchmark_s2():
        for i in range(rounds):
            s2 = s2 * mul
    @parameter
    fn _benchmark_s3():
        for i in range(rounds):
            s3 = s3 * mul

    keep(s1)
    keep(s2)
    keep(s3)

    var b1_mean = benchmark.run[_benchmark_s1]().mean("ms")
    var b2_mean = benchmark.run[_benchmark_s2]().mean("ms")
    var b3_mean = benchmark.run[_benchmark_s3]().mean("ms")

    print("S1 (tensor based):",b1_mean,"ms")
    print("S2 (dtypepointer):",b2_mean,"ms")
    print("S3 (dtypepointer/register_passable):",b3_mean,"ms")
    print("\nS1/S2:",b1_mean/b2_mean)
    print("S1/S3:",b1_mean/b3_mean)
    print("S2/S3:",b2_mean/b3_mean)
