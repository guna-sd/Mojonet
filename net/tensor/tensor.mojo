from .tutils import shape, Tensorprinter
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from collections.optional import Optional, Variant
from net.kernel import scalar_op, tensor_op, Broadcast_op, vectorize, parallelize, calculate_shapes, matmul, randn

@value
struct Tensor[type : DType]:
    """
    A tensor is a multi-dimensional array of elements.
    """
    var shape : shape
    """
    The shape is representing the dimensions of the tensor.
    """
    var dtype : DType
    """
    The data type of the tensor is a type that defines the type of the elements in the tensor.
    """
    var storage : DTypePointer[type]
    """
    The storage is a pointer to a block of memory that holds the elements of the tensor.
    """

    fn __init__(inout self):
        self.storage = DTypePointer[type]().alloc(0)
        self.shape = shape()
        self.dtype = type

    fn __init__(inout self, *shapes : Int):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)
    
    fn __init__(inout self, data : DTypePointer[type], shape : shape):
        self.storage = data
        self.dtype = type
        self.shape = shape

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
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor storage at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to load.
        
        Args:
            index : The index in the tensor storage from which to load the SIMD value. If negative, it is interpreted as an index from the end of the storage.

        Returns:
            The SIMD value loaded from the tensor storage at the specified index.
        """
        if index < 0:
            index =  self.num_elements() + index
        return self.storage.load[width=nelts](index)
    
    fn store[nelts : Int](self, owned index: Int, value: SIMD[type,nelts]):
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor storage at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to store.
        
        Args:
            index : The index in the tensor storage at which to store the SIMD value. If negative, it is interpreted as an index from the end of the storage.
            value : The SIMD value to store in the tensor storage at the specified index.
        """
        if index < 0:
            index = self.num_elements() + index
        self.storage.store[width=nelts](index, value)

    fn load(self, index : Int) -> SIMD[type,1]:
        return self.load[1](index)
    
    fn load(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)

    fn store(self: Self, index: Int, val: SIMD[type, 1]):
        self.store[1](index, val)

    fn store(self: Self, *indices: Int, val: SIMD[type, 1]):
        var pos = self.shape.offset(indices)
        self.store(pos, val)

    fn store(self: Self, indices: List[Int], val: SIMD[type, 1]):
        var pos = self.shape.offset(indices)
        self.store(pos, val)

    fn __getitem__(self: Self, index: Int)-> SIMD[type,1]:
        return self.load[1](index)
    
    fn __getitem__(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)

    fn __getitem__(self, indices : List[Int]) -> SIMD[type,1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)

    fn __setitem__(self: Self, index: Int, val: SIMD[type, 1]):
        self.store(index, val)

    fn __setitem__(self: Self, indices: List[Int], val: SIMD[type, 1]):
        var pos = self.shape.offset(indices)
        self[pos] = val
    
    fn __eq__(self: Self, other: Self) -> Bool:
        var val = False
        if self.num_elements() == other.num_elements() and self.shape == other.shape:
            for i in range(self.num_elements()):
                if self.storage[i] == other.storage[i]:
                    val = True
        return val
    
    fn __eq__(self : Self, other: _Tensor[type]) -> Bool:
        var val = False
        if self.num_elements() == other.num_elements() and self.shape == other.shape():
            for i in range(self.num_elements()):
                if self[i] == other[i]:
                    val = True
        return val

    fn __ne__(self: Self, other: Self) -> Bool:
        if self.__eq__(other):
            return False
        return True

    fn __ne__(self: Self, other: _Tensor[type]) -> Bool:
        if self.__eq__(other):
            return False
        return True

    fn __add__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.add](self,other)
        return self

    fn __add__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.add](self,other)
            
    fn __sub__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.sub](self,other)
        return self

    fn __sub__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.sub](self,other)
    
    fn __mul__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.mul](self,other)
        return self

    fn __mul__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.mul](self,other)
    
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.div](self,other)
        return self

    fn __truediv__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.div](self,other)
    
    fn __pow__(self: Self, exponent: Int) -> Self:
        for i in range(self.shape.num_elements):
            self.storage[i] = math.pow(self.storage[i], exponent)
        return self

    @always_inline
    fn __matmul__(self: Self, other: Self) -> Self:
        """
        Implements matrix multiplication for Tensor.
        The operation is defined as self @ other.
        """
        return matmul(self,other)
    
    fn __enter__(owned self) -> Self:
        """The function to call when entering the context."""
        return self ^

    fn __str__(self: Self) -> String:
        return Tensorprinter(self.storage, self.shape)
   
    @always_inline
    fn pow(self: Self, pow: Int):    
        for i in range(self.num_elements()):
            self.storage[i] = self.storage[i] ** pow
    
    @always_inline
    fn add(self : Self, x : SIMD[type,1]) -> Self:
        return self.__add__(x)

    @always_inline
    fn add(self: Self, other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__truediv__(other)

        return Broadcast_op[type,math.add](self,other)

    @always_inline
    fn sub(self : Self, x : SIMD[type,1]) -> Self:
        return self.__sub__(x)

    @always_inline
    fn sub(self: Self, other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__sub__(other)

        return Broadcast_op[type,math.sub](self,other)

    @always_inline
    fn multiply(self : Self, x : SIMD[type,1]) -> Self:
        return self.__mul__(x)

    @always_inline
    fn multiply(self: Self, other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__mul__(other)

        return Broadcast_op[type,math.mul](self,other)

    @always_inline
    fn div(self : Self, x : SIMD[type,1]) -> Self:
        return self.__truediv__(x)

    fn div(self: Self, other: Tensor[type]) -> Self:
        if self.shape == other.shape:
            return self.__truediv__(other)

        return Broadcast_op[type,math.div](self,other)

    @always_inline
    fn zeros(self : Self):
        memset_zero(self.storage, self.shape.num_elements)
    
    @always_inline
    fn ones(self : Self):
        for i in range(self.shape.num_elements):
            self.storage.store(i,1)

    @always_inline
    fn rand(self, seed : Optional[Int] = None):
        if seed:
            random.seed(seed.value())
        else:
            random.seed()
        random.randn(self.storage, self.num_elements(),0,self.rank())
            
    @always_inline
    fn random(self, seed : Optional[Int]) -> Self:
        if seed:
            random.seed(seed.value())
        else:
            random.seed()
        random.randn(self.storage, self.num_elements(),0,self.rank())
        return self

    fn fill(self: Self, value: SIMD[type, 1]):
        """ Fill the tensor with a specified value."""
        for i in range(self.num_elements()):
            self.storage[i] = value

    @always_inline
    fn transposed(self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))
            abort(external_call["exit", Int](1))

        var tshape = self.shape._shapelist  
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        var ttensor = Tensor[type](tshape)
        
        for index in range(self.num_elements()):
            var _indices = self.shape.indices(index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            ttensor[tindices] = self[index]
        
        return ttensor
        
    @always_inline
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var ttensor = self.transposed(dim1, dim2)
        self = Self(ttensor.storage, ttensor.shape)

    @always_inline
    fn broadcast(self: Self, shapes: shape) -> Self:
        """
        Broadcasts the tensor to a specified shape and returns a new tensor with the broadcasted shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        return Broadcast_op[type, math.add](self, Tensor[type](shapes))

    @always_inline
    fn broadcast_to(inout self: Self, shapes: shape):
        """
        Broadcasts the tensor to a specified shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        var result = self.broadcast(shapes)
        self = Self(result.storage, result.shape)

    @always_inline
    fn reshape(self: Self, other: shape) -> Self:
        """ Reshape the tensor to the new dimensions and returns the reshaped tensor."""
        if self.shape._rank != other._rank or self.shape.num_elements != other.num_elements:
            print("Error: Cannot reshape tensor.")
            abort(external_call["exit", Int](1))

        var data = Tensor[type](other)
        for idx in range(self.shape.num_elements):
            var old_indices = self.shape.indices(idx)
            var new_indices = other.indices(idx)
            data[new_indices] = self[old_indices]
        return Self(data.storage, other)

    @always_inline
    fn reshape_to(inout self: Self, other: shape):
        """ Reshape the tensor to the new dimensions."""
        self = Self(self.reshape(other).storage, other)

    @always_inline
    fn rank(self: Self) -> Int:
        return self.shape._rank

    @always_inline
    fn _shape(self: Self) ->shape:
        return self.shape
        
    @always_inline
    fn _dtype(self: Self) -> String:
        return self.dtype.__str__()
    
    @always_inline
    fn num_elements(self: Self) -> Int:
        return self.shape.num_elements
    
    fn astype[dtype : DType](self : Self) -> Tensor[dtype]:
        var casted = Tensor[dtype](self.shape)
        alias nelts = simdwidthof[dtype]()
        var num_elements = self.shape.num_elements

        @parameter
        fn caster(start_index: Int):
            @parameter
            fn cast_single_element[nelts : Int](index: Int):
                casted.store[nelts](start_index + index, self[start_index + index].cast[dtype]())

            vectorize[cast_single_element, nelts](num_elements - start_index)

        parallelize[caster](num_elements, nelts)
        return casted

    @always_inline   
    fn num_bytes(self: Self) -> Int:
        return sizeof[type]() * self.shape.num_elements
    
    @always_inline  
    fn itemsize(self: Self) -> Int:
        return sizeof[type]()

fn tensor[dtype : DType = DType.int8](Shape : List[Int], rand : Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor

fn tensor[dtype : DType = DType.int8](*Shape : Int, rand : Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor

fn tensor[dtype : DType = DType.int8](Shape : VariadicList[Int], rand : Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor

fn ones[dtype : DType = DType.int8](Shape : VariadicList[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.ones()
    return tensor

fn ones[dtype : DType = DType.int8](Shape : List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.ones()
    return tensor

fn ones[dtype : DType = DType.int8](*Shape : Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.ones()
    return tensor

fn zeros[dtype : DType = DType.int8](Shape : VariadicList[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.zeros()
    return tensor

fn zeros[dtype : DType = DType.int8](Shape : List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.zeros()
    return tensor

fn zeros[dtype : DType = DType.int8](*Shape : Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.zeros()
    return tensor

fn fill[dtype : DType = DType.int8](*Shape : Int, rand : Bool = True) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.ones()
    return tensor

fn fill[dtype : DType = DType.int8](Shape : List[Int], rand : Bool = True) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.ones()
    return tensor

fn fill[dtype : DType = DType.int8](Shape : VariadicList[Int], rand : Bool = True) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.ones()
    return tensor

fn rand[dtype : DType = DType.int8](Shape : List[Int], seed : Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand(seed)
    tensor.rand()
    return tensor

fn rand[dtype : DType = DType.int8](Shape : VariadicList[Int], seed : Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand(seed)
    tensor.rand()
    return tensor

fn rand[dtype : DType = DType.int8](*Shape : Int, seed : Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand(seed)
    tensor.rand()
    return tensor