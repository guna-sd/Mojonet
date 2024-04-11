from net.utils import shape, Tensorprinter, _bytes
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from random.random import rand
from net.nn import gelu



@value
struct Tensor[type : DType]:
    var shape : shape
    var dtype : DType
    var storage : DTypePointer[type]

    fn __init__(inout self):
        self.storage = DTypePointer[type]()
        self.shape = shape()
        self.dtype = type

    fn __init__(inout self, *shapes : Int):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)

    fn __init__(inout self, shapes : VariadicList[Int]):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)

    
    fn __init__(inout self, shapes : List[Int]):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)

    
    fn __init__[size : Int](inout self, shapes : StaticIntTuple[size]):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)


    fn __init__(inout self : Self, shapes : shape):
        self.shape = shapes
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)


    fn __init__(inout self : Self, shapes : TensorShape):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)


    fn __init__(inout self : Self, shapes : TensorSpec):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)

    
    fn __init__(inout self : Self, data : _Tensor[type]):
        self.shape = shape(data.shape())
        self.storage = data._ptr
        self.dtype = type

    fn __init__(inout self : Self, shapes : shape, *data : SIMD[type,1]):
        self.shape = shapes
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        for i in range(data.__len__()):
            self.storage[i] = data[i]
        self.dtype = type

    fn __init__(inout self : Self, shapes : List[Int], data : List[SIMD[type, 1]]):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        for i in range(len(data)):
            self.storage[i] = data[i]
        self.dtype = type

    fn __init__[size : Int](inout self : Self, shapes : StaticIntTuple[size], data : StaticIntTuple[size]):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        for i in range(len(data)):
            self.storage[i] = data[i]
        self.dtype = type

    fn __init__(inout self : Self, shapes : VariadicList[Int], data : VariadicList[SIMD[type, 1]]):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        for i in range(len(data)):
            self.storage[i] = data[i]
        self.dtype = type

    fn __del__(owned self):
        self.storage.free()

    fn __copyinit__(inout self: Self, other: Self):
        self.shape = other.shape
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memcpy(self.storage, other.storage, self.shape.num_elements)
        self.dtype = other.dtype
    
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape
        self.storage = existing.storage
        self.dtype = existing.dtype
    
    fn __getitem__(self, index : Int) -> SIMD[type, 1]:
        return self.storage.load(index)
    
    
    fn __setitem__(self, index : Int, value : SIMD[type, 1]):
        self.storage[index] = value
    
    fn __eq__(self: Self, other: Self) -> Bool:
        var bool = False
        for i in range(self.shape.num_elements):
            if self.storage[i] == other.storage[i]: 
                bool = True
            else:
                bool = False
        return bool

    fn __ne__(self: Self, other: Self) -> Bool:
        if self.storage == other.storage:
            return False
        return True
    
    fn __len__(self: Self) -> Int:
        return self.shape.num_elements

    fn __add__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
            if self.shape.num_elements == other.shape.num_elements:
                if self.dtype == other.dtype:
                    if self.shape.shape == other.shape.shape:
                        for i in range(self.shape.num_elements):
                            self.storage[i] += other.storage[i]
        return self
    
    fn __sub__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
            if self.shape.num_elements == other.shape.num_elements:
                if self.dtype == other.dtype:
                    if self.shape.shape == other.shape.shape:
                        for i in range(self.shape.num_elements):
                            self.storage[i] -= other.storage[i]
        return self
    
    fn __mul__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
            if self.shape.num_elements == other.shape.num_elements:
                if self.dtype == other.dtype:
                    if self.shape.shape == other.shape.shape:
                        for i in range(self.shape.num_elements):
                            self.storage[i] *= other.storage[i]
        return self
    
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
            if self.shape.num_elements == other.shape.num_elements:
                if self.dtype == other.dtype:
                    if self.shape.shape == other.shape.shape:
                        for i in range(self.shape.num_elements):
                            self.storage[i] /= other.storage[i]
        return self
    
    fn __pow__(self: Self, exponent: Int) -> Self:
        for i in range(self.shape.num_elements):
            self.storage[i] = math.pow(self.storage[i], exponent)
        return self

    fn load[nelts : Int](self, index: Int) -> SIMD[type,nelts]:
        return self.storage.load[width=nelts](index)
    
    fn store[nelts : Int](self, index: Int, value: SIMD[type,nelts]):
        self.storage.store[width=nelts](index, value)    


    # fn reshape(inout self: Self, shapes: List[Int]):
    #     self.shape = shape(shapes)

    fn __str__(self: Self) -> String:
        return Tensorprinter(self.storage, self.shape)
    
    
    @always_inline("nodebug")
    fn pow(self: Self, pow: Int):    
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] ** pow

    @always_inline("nodebug")
    fn sum(self : Self, x : Tensor[type]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] + x.storage[i]
    
    @always_inline("nodebug")
    fn sum(self : Self, x : SIMD[type,1]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] + x

    @always_inline("nodebug")
    fn sub(self : Self, x : SIMD[type,1]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] - x

    @always_inline("nodebug")
    fn sub(self : Self, x : Tensor[type]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] - x.storage[i]

    @always_inline("nodebug")
    fn multiply(self : Self, x : SIMD[type,1]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] * x

    @always_inline("nodebug")
    fn multiply(self : Self, x : Tensor[type]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] * x.storage[i]    

    @always_inline("nodebug")
    fn div(self : Self, x : SIMD[type,1]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] / x

    @always_inline("nodebug")
    fn div(self : Self, x : Tensor[type]):
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] / x.storage[i]

    @always_inline("nodebug")
    fn zeros(inout self : Self):
        memset_zero(self.storage, self.shape.num_elements)
    
    @always_inline("nodebug")
    fn ones(inout self : Self):
        for i in range(self.shape.num_elements):
            self.storage.store(i,1)

    @always_inline("nodebug")
    fn rand(inout self):
        rand(self.storage, self.num_elements())

    @always_inline("nodebug")
    fn rank(self: Self) -> Int:
        return self.shape.rank()

    @always_inline("nodebug")
    fn _shape(self: Self) ->shape:
        return self.shape
        
    @always_inline("nodebug")
    fn _dtype(self: Self) -> String:
        return self.dtype.__str__()
    
    @always_inline("nodebug")
    fn num_elements(self: Self) -> Int:
        return self.shape.count_elements()

    @always_inline("nodebug")    
    fn num_bytes(self: Self) -> Int:
        return _bytes(self.num_elements(), self.dtype)
    
    @always_inline("nodebug")    
    fn itemsize(self: Self) -> Int:
        return _bytes(1,self.dtype)
    
    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self ^
    
    fn to(self: Self, device: String):
        if device == "cpu":
            print("cpu")
        elif device == "cuda":
            print("cuda")
        else:
            print("unknown device")

fn main():
    var x = Tensor[DType.float32](2,2)
    x[0] = 0.1315377950668335
    x[1] = 0.458650141954422
    x[2] = 1.21895918250083923
    x[3] = 0.67886471748352051
    print(x)
    var F = gelu[DType.float32,4,'v'](x)
    print(F)