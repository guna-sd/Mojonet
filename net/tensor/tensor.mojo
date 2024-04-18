from .tutils import shape, Tensorprinter, _bytes
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from random.random import rand
from algorithm import vectorize, parallelize
from sys.info import num_physical_cores


@always_inline("nodebug")
fn bin_ops[dtype : DType, func : fn[type: DType, simd_width: Int](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[type, simd_width]](
    Input1 : Tensor[dtype], Input2 : Tensor[dtype]) -> Tensor[dtype]:
    var shape = Input1.shape == Input2.shape
    var elm = Input1.num_elements() == Input2.num_elements()
    if shape != elm: 
        print(Error("Both inputs must be the same shape"))
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()
    var Output = Tensor[dtype](Input1.shape)

    @parameter
    fn calc(i : Int):

        @parameter
        fn operation[nelts : Int](j : Int):
            Output.store(j, func[dtype,nelts](Input1[j], Input2[j]))
        vectorize[operation, nelts](Input1.num_elements())
    parallelize[calc](Input1.num_elements(), num_cores)
    
    return Output


@always_inline("nodebug")
fn bin_ops[dtype : DType, func : fn[type: DType, simd_width: Int](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[type, simd_width]](
    Input : Tensor[dtype], value : SIMD[dtype,1]) -> Tensor[dtype]:
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()
    var Output = Tensor[dtype](Input.shape)

    @parameter
    fn calc(i : Int):

        @parameter
        fn operation[nelts : Int](j : Int):
            Output.store(j, func[dtype,nelts](Input[j], value))
        vectorize[operation, nelts](Input.num_elements())
    parallelize[calc](Input.num_elements(), num_cores)
    
    return Output


@value
struct Tensor[type : DType]:
    var shape : shape
    var dtype : DType
    var storage : DTypePointer[type]

    fn __init__(inout self):
        self.storage = DTypePointer[type]().alloc(0)
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
        self.shape.shape.free()

    fn __copyinit__(inout self: Self, other: Self):
        self.shape = other.shape
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memcpy(self.storage, other.storage, self.shape.num_elements)
        self.dtype = other.dtype
    
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape
        self.storage = existing.storage
        self.dtype = existing.dtype

    fn load[nelts : Int](self, owned index: Int) -> SIMD[type,nelts]:
        if index < 0:
            index =  self.num_elements() + index
        return self.storage.load[width=nelts](index)
    
    fn store[nelts : Int](self, owned index: Int, value: SIMD[type,nelts]):
        if index < 0:
            index = self.num_elements() + index
        self.storage.store[width=nelts](index, value)

    fn __getitem__(self, owned offset : Int) -> SIMD[type,1]:
        return self.load[1](offset)
    
    fn __getitem__(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.position(indices)
        return self.load[1](pos)
    
    fn __getitem__(self: Self, indices: VariadicList[Int]) -> SIMD[type, 1]:
        var pos = self.shape.position(indices)
        return self.load[1](pos)
    
    fn __getitem__(self: Self, indices : List[Int]) -> SIMD[type,1]:
        var pos = self.shape.position(indices)
        return self.load[1](pos)
    
    fn __setitem__(self, owned offset : Int, value : SIMD[type,1]):
        self.store(offset, value)
    
    fn __setitem__(self: Self, indices: VariadicList[Int], val: SIMD[type, 1]):
        var pos = self.shape.position(indices)
        self.store(pos, val)
    
    fn __setitem__(self: Self, *indices: Int, val: SIMD[type,1]):
        var pos = self.shape.position(indices)
        self.store(pos, val)
    
    fn __setitem__(self: Self, indices : List[Int], val: SIMD[type,1]):
        var pos = self.shape.position(indices)
        self.store(pos, val)
    
    fn __eq__(self: Self, other: Self) -> Bool:
        var equal = True

        @parameter
        fn compare[ints : Int](i : Int):
            if self.storage[i] == other.storage[i]:
                equal = True
            equal = False
        vectorize[compare,0](self.shape.num_elements)

        return equal

    fn __ne__(self: Self, other: Self) -> Bool:
        if self.__eq__(other):
            return False
        return True

    fn __add__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return bin_ops[type,math.add](self,other)
        return self
    
    fn __sub__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return bin_ops[type,math.sub](self,other)
        return self
    
    fn __mul__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return bin_ops[type,math.mul](self,other)
        return self
    
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return bin_ops[type,math.div](self,other)
        return self
    
    fn __pow__(self: Self, exponent: Int) -> Self:
        for i in range(self.shape.num_elements):
            self.storage[i] = math.pow(self.storage[i], exponent)
        return self

    fn reshape(inout self: Self, *shapes: Int):
        self.shape = shape(shapes)

    fn __str__(self: Self) -> String:
        return Tensorprinter(self.storage, self.shape)
   
    @always_inline("nodebug")
    fn pow(self: Self, pow: Int):    
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] ** pow

    @always_inline("nodebug")
    fn add(inout self : Self, x : Tensor[type]):
        self = bin_ops[type,math.add](self,x)
    
    @always_inline("nodebug")
    fn add(inout self : Self, x : SIMD[type,1]):
        self = bin_ops[type,math.add](self,x)
            
    fn sum(self : Self, other : Self) -> Self:
        return self.__add__(other)

    @always_inline("nodebug")
    fn sub(inout self : Self, x : SIMD[type,1]):
        self = bin_ops[type,math.sub](self,x)

    @always_inline("nodebug")
    fn sub(inout self : Self, x : Tensor[type]):
        self = bin_ops[type,math.sub](self,x)

    @always_inline("nodebug")
    fn multiply(inout self : Self, x : SIMD[type,1]):
        self = bin_ops[type,math.mul](self,x)

    @always_inline("nodebug")
    fn multiply(inout self : Self, x : Tensor[type]):
        self = bin_ops[type,math.mul](self,x)   

    @always_inline("nodebug")
    fn div(inout self : Self, x : SIMD[type,1]):
        self = bin_ops[type,math.div](self,x) 

    @always_inline("nodebug")
    fn div(inout self : Self, x : Tensor[type]):
        self = bin_ops[type,math.div](self,x) 

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
    fn random(inout self) -> Self:
        rand(self.storage, self.num_elements())
        return self

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
    fn transpose(inout self):
        ...
        
    @always_inline("nodebug")    
    fn num_bytes(self: Self) -> Int:
        return _bytes(self.num_elements(), self.dtype)
    
    @always_inline("nodebug")    
    fn itemsize(self: Self) -> Int:
        return _bytes(1,self.dtype)
    
    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self ^