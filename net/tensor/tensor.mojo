from .tutils import shape, Tensorprinter
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from collections.optional import Optional, Variant
from net.kernel import scalar_op, tensor_op, Broadcast_op, vectorize, parallelize, calculate_shapes, matmul, randn, num_physical_cores

@value
struct Tensor[type : DType = DType.float32]:
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

    fn __init__(inout self, *shapes : Int,):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)      
    
    fn __init__(inout self, shape : shape, data : DTypePointer[type],):
        self.storage = data
        self.dtype = type
        self.shape = shape

    fn __init__(inout self, shape : shape, data : DTypePointer[type], value : Int,):
        self.storage = data
        self.dtype = type
        self.shape = shape
        self = self.fill(value)

    fn __init__(inout self, shapes : VariadicList[Int],):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)
    
    fn __init__(inout self, shapes : List[Int],):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)
    
    fn __init__[size : Int](inout self, shapes : StaticIntTuple[size],):
        self.shape = shape(shapes)
        self.dtype = type
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.storage, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : shape,):
        self.shape = shapes
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : TensorShape,):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : TensorSpec,):
        self.shape = shape(shapes)
        self.storage = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.storage, self.shape.num_elements)
    
    fn __init__(inout self : Self, data : _Tensor[type],):
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
        return tensor_op[type,math.add](self,other)

    fn __add__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.add](self,other)

    fn __radd__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.add](self,other)

    fn __iadd__(inout self: Self, other: Self):
        self = tensor_op[type,math.add](self,other)
 
    fn __iadd__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.add](self,other)

    fn __sub__(self: Self, other: Self) -> Self:
        return tensor_op[type,math.sub](self,other)

    fn __sub__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.sub](self,other)

    fn __rsub__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.sub](self,other)

    fn __isub__(inout self: Self, other: Self):
        self = tensor_op[type,math.sub](self,other)
 
    fn __isub__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.sub](self,other)
    
    fn __mul__(self: Self, other: Self) -> Self:
        return tensor_op[type,math.mul](self,other)

    fn __mul__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.mul](self,other)

    fn __rmul__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.mul](self,other)

    fn __imul__(inout self: Self, other: Self):
        self = tensor_op[type,math.mul](self,other)
 
    fn __imul__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.mul](self,other)
    
    fn __truediv__(self: Self, other: Self) -> Self:
        return tensor_op[type,math.div](self,other)

    fn __truediv__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.div](self,other)

    fn __rtruediv__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.div](self,other)

    fn __itruediv__(inout self: Self, other: Self):
        self = tensor_op[type,math.div](self,other)
 
    fn __itruediv__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.div](self,other)
    
    fn __pow__(self: Self, exponent: Int) -> Self:
        """
        Exponentiation of each element in the tensor by the given exponent.
        """
        var result = self
        for i in range(result.shape.num_elements):
            result.storage[i] = math.pow(result.storage[i], exponent)
        return result

    fn __ipow__(inout self: Self, exponent: Int):
        """
        In-place exponentiation of each element in the tensor by the given exponent.
        """
        for i in range(self.shape.num_elements):
            self.storage[i] = math.pow(self.storage[i], exponent)


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
    fn sum(self: Self) -> Scalar[type]:
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]()
        @parameter
        fn _sum[nelts : Int](i : Int):
            result += self[i]
        vectorize[_sum,nelts](self.num_elements())
        return result

    @always_inline
    fn max(self: Self) -> Scalar[type]:
        """
        Find the maximum value in the tensor.

        Returns:
            The maximum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]()
        @parameter
        fn _max[nelts : Int](i : Int):
            result = math.max(result, self[i])
        vectorize[_max,nelts](self.num_elements())
        return result

    @always_inline
    fn min(self: Self) -> Scalar[type]:
        """
        Find the minimum value in the tensor.

        Returns:
            The minimum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = simdwidthof[type]()
        @parameter
        fn _min[nelts : Int](i : Int):
            result = math.min(result, self[i])
        vectorize[_min,nelts](self.num_elements())
        return result

    @always_inline
    fn mean(self: Self) -> Scalar[type]:
        """
        Compute the mean (average) value of the tensor.

        Returns:
            The mean value of the tensor as a scalar value.
        """
        return self.sum() / Scalar[type](self.num_elements())

    @always_inline
    fn prod(self: Self) -> Scalar[type]:
        """
        Compute the product of all elements in the tensor.

        Returns:
            The product of all elements in the tensor as a scalar value.
        """
        var result = Scalar [type]()
        alias nelts = simdwidthof[type]()
        @parameter
        fn _prod[nelts : Int](i : Int):
            result *= self[i]
        vectorize[_prod,nelts](self.num_elements())
        return result

    @always_inline
    fn arange(self, start: Scalar[type], end: Scalar[type], step: Scalar[type] = 1) -> Tensor[type]:
        """
        Returns a tensor with values from start to end with specified step size.

        Args:
            start: The start value of the sequence.
            end: The end value of the sequence.
            step: The step size between consecutive values. Default is 1.

        Returns:
            A tensor containing the values from start to end with the specified step size.
        """
        var result = Tensor[type](self.shape)
        var value = start
        for i in range(self.num_elements()):
            result[i] = value
            value += step
        return result
    
    @always_inline
    fn arange(inout self):
        self = self.arange(0,self.num_elements())

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
            random.seed(seed.value()[])
        else:
            random.seed()
        random.randn(self.storage, self.num_elements(),0,self.rank())
            
    @always_inline
    fn random(self, seed : Optional[Int]) -> Self:
        if seed:
            random.seed(seed.value()[])
        else:
            random.seed()
        random.randn(self.storage, self.num_elements(),0,self.rank())
        return self

    fn fill(self: Self, value: Scalar[type]) -> Self:
        """Fill the tensor with a specified value."""
        var result = self
        var num_elements = self.shape.num_elements
        alias nelts = simdwidthof[type]()

        @parameter
        fn filler(start_index: Int):
            @parameter
            fn _set[nelts : Int](index: Int):
                result.store[nelts](start_index + index, value)

            vectorize[_set, nelts](num_elements - start_index)

        parallelize[filler](num_elements, nelts)
        return result

    fn ifill(self: Self, value: Scalar[type]):
        """Fill the tensor with a specified value."""
        var num_elements = self.shape.num_elements
        alias nelts = simdwidthof[type]()

        @parameter
        fn filler(start_index: Int):
            @parameter
            fn _set[nelts : Int](index: Int):
                self.store[nelts](start_index + index, value)

            vectorize[_set, nelts](num_elements - start_index)

        parallelize[filler](num_elements, nelts)

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
        self = Self(ttensor.shape, ttensor.storage)

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
        self = Self(result.shape, result.storage)

    @always_inline
    fn reshape(self: Self, other: shape) -> Self:
        """ Reshape the tensor to the new dimensions and returns the reshaped tensor."""
        if self.shape.num_elements != other.num_elements:
            print("Error: Cannot reshape tensor.")
            abort(external_call["exit", Int](1))

        var data = Tensor[type](other)
        for idx in range(self.shape.num_elements):
            var old_indices = self.shape.indices(idx)
            var new_indices = other.indices(idx)
            data[new_indices] = self[old_indices]
        return Self(other, data.storage)

    @always_inline
    fn reshape(self: Self, *new: Int) -> Self:
        """
        Reshape the tensor to the specified new shape.
        
        Args:
            new: The new shape dimensions as variadic integers.

        Returns:
            A new tensor reshaped to the new dimensions specified.
        """   
        return self.reshape(shape(new))

    @always_inline
    fn reshape_to(inout self: Self, other: shape):
        """ Reshape the tensor to the new dimensions."""
        self = Self(other, self.reshape(other).storage)

    @always_inline
    fn flatten(self: Self) -> Self:
        """
        Flatten the tensor into a 1D array.
        
        Returns:
            A new tensor with all elements in a single dimension.
        """
        var new_shape = shape(self.shape.num_elements)
        return self.reshape(new_shape)

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