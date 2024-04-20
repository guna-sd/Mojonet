from .tutils import shape, Tensorprinter, _bytes, indices, flatten_index, __get_position
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from random.random import rand as _rand
from net.kernel import scalar_op, tensor_op, vectorize, matmul


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
        var pos = self.shape.offset(indices(self.shape._shapelist,offset))
        return self.load[1](pos)
    
    fn __getitem__(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)
    
    fn __getitem__(self: Self, indices: VariadicList[Int]) -> SIMD[type, 1]:
        var pos = self.shape.offset(indices)
        return self.load[1](pos)
    
    fn __setitem__(self, owned offset : Int, value : SIMD[type,1]):
        var pos = self.shape.offset(indices(self.shape._shapelist,offset))
        self.store(pos, value)
    
    fn __setitem__(self: Self, indices: VariadicList[Int], val: SIMD[type, 1]):
        var pos = self.shape.offset(indices)
        self.store(pos, val)
    
    fn __setitem__(self: Self, *indices: Int, val: SIMD[type,1]):
        var pos = self.shape.offset(indices)
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
                    return tensor_op[type,math.add](self,other)
        return self
            
    fn __sub__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.sub](self,other)
        return self
    
    fn __mul__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.mul](self,other)
        return self
    
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.shape._rank == other.shape._rank:
                if self.dtype == other.dtype:
                    return tensor_op[type,math.div](self,other)
        return self
    
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
        return matmul[type](self, other)
    
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
    fn add(inout self : Self, x : SIMD[type,1]):
        self = scalar_op[type,math.add](self,x)

    @always_inline
    fn add(inout self: Self, inout other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__add__(other)

        var broadcasted_shape = self.shape.broadcast_shapes(other.shape)
        if broadcasted_shape.rank() == 0:
            print(Error("Cannot add tensors with incompatible shapes"))
            return self

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)
            var other_indices  = indices(other.shape._shapelist, idx)

            for j in range(self.shape.rank()):
                if result.shape[j] == 1:
                    result_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = flatten_index(self.shape, result_indices)
            var other_idx = flatten_index(other.shape, other_indices)

            result[idx] = self[self_idx] + other[other_idx]
        return result

    @always_inline
    fn sub(inout self : Self, x : SIMD[type,1]):
        self = scalar_op[type,math.sub](self,x)

    @always_inline
    fn sub(inout self: Self, inout other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__sub__(other)

        var broadcasted_shape = self.shape.broadcast_shapes(other.shape)
        if broadcasted_shape.rank() == 0:
            print(Error("Cannot add tensors with incompatible shapes"))
            return self

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)
            var other_indices  = indices(other.shape._shapelist, idx)

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    result_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = flatten_index(self.shape, result_indices)
            var other_idx = flatten_index(other.shape, other_indices)

            result[idx] = math.sub(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn multiply(inout self : Self, x : SIMD[type,1]):
        self = scalar_op[type,math.mul](self,x)

    @always_inline
    fn multiply(inout self: Self, inout other: Tensor[type]) -> Self:

        if self.shape.num_elements == other.shape.num_elements:
            return self.__mul__(other)

        var broadcasted_shape = self.shape.broadcast_shapes(other.shape)
        if broadcasted_shape.rank() == 0:
            print(Error("Cannot add tensors with incompatible shapes"))
            return self

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)
            var other_indices  = indices(other.shape._shapelist, idx)

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    result_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = flatten_index(self.shape, result_indices)
            var other_idx = flatten_index(other.shape, other_indices)

            result[idx] = math.mul(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn div(inout self : Self, x : SIMD[type,1]):
        self = scalar_op[type,math.div](self,x) 

    @always_inline
    fn div(inout self: Self, inout other: Tensor[type]) -> Self:

        if self.shape == other.shape:
            return self.__truediv__(other)

        var broadcasted_shape = self.shape.broadcast_shapes(other.shape)
        if broadcasted_shape.rank() == 0:
            print(Error("Cannot add tensors with incompatible shapes"))
            return self

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)
            var other_indices  = indices(other.shape._shapelist, idx)

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    result_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = flatten_index(self.shape, result_indices)
            var other_idx = flatten_index(other.shape, other_indices)

            result[idx] = math.div(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn zeros(inout self : Self):
        memset_zero(self.storage, self.shape.num_elements)
    
    @always_inline
    fn ones(inout self : Self):
        for i in range(self.shape.num_elements):
            self.storage.store(i,1)

    @always_inline
    fn rand(inout self):
        _rand(self.storage, self.num_elements())

    @always_inline
    fn random(inout self) -> Self:
        _rand(self.storage, self.num_elements())
        return self

    fn fill(inout self: Self, value: SIMD[type, 1]):
        """ Fill the tensor with a specified value."""
        for i in range(self.num_elements()):
            self.storage[i] = value

    @always_inline
    fn transposed(self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:

        var _shape = self.shape._shapelist        
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))

        var tshape = _shape
        
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        
        var ttensor = Tensor[type](tshape)
        
        for index in range(self.num_elements()):
            var _indices = indices(_shape, index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            var tindex = ttensor.shape.position(tindices)
            ttensor[tindex] = self[index]
        
        return ttensor
        
    @always_inline
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var _shape = self.shape._shapelist        
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))

        var tshape = _shape
        
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        
        var ttensor = Tensor[type](tshape)
        
        for index in range(self.num_elements()):
            var _indices = indices(_shape, index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            var tindex = ttensor.shape.position(tindices)
            ttensor[tindex] = self[index]
        
        self = ttensor

    @always_inline
    fn broadcast(inout self: Self, shapes: shape) -> Self:
        """
        Broadcasts the tensor to a specified shape and returns a new tensor with the broadcasted shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        var broadcasted_shape = self.shape.broadcast_shapes(shapes)
        
        if broadcasted_shape.rank() == 0:
            print("Cannot broadcast tensor due to incompatible shapes")

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)            
            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    result_indices[j] = 0

            var _idx = self.shape.offset(result_indices)

        return result

    @always_inline
    fn broadcast_to(inout self: Self, shapes: shape):
        """
        Broadcasts the tensor to a specified shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        var broadcasted_shape = self.shape.broadcast_shapes(shapes)
        
        if broadcasted_shape.rank() == 0:
            print("Cannot broadcast tensor due to incompatible shapes")

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)            
            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    result_indices[j] = 0

            var _idx = self.shape.offset(result_indices)

            result[idx] = self[_idx]
        self = result

    @always_inline
    fn reshape(inout self: Self, new_shapes: List[Int]):
        """ Reshape the tensor to the new dimensions."""
        var total = 1
        for i in range(new_shapes.__len__()):
            total *= i
        if not total == self.num_elements():
            print("New shape must contain the same number of elements as the original.")
        self.shape = shape(new_shapes)

    @always_inline
    fn rank(self: Self) -> Int:
        return self.shape.rank()

    @always_inline
    fn _shape(self: Self) ->shape:
        return self.shape
        
    @always_inline
    fn _dtype(self: Self) -> String:
        return self.dtype.__str__()
    
    @always_inline
    fn num_elements(self: Self) -> Int:
        return self.shape.count_elements()

    @always_inline   
    fn num_bytes(self: Self) -> Int:
        return _bytes(self.num_elements(), self.dtype)
    
    @always_inline  
    fn itemsize(self: Self) -> Int:
        return _bytes(1,self.dtype)

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

fn rand[dtype : DType = DType.int8](Shape : List[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.rand()
    return tensor

fn rand[dtype : DType = DType.int8](Shape : VariadicList[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.rand()
    return tensor

fn rand[dtype : DType = DType.int8](*Shape : Int) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    tensor.rand()
    return tensor