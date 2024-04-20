from .tutils import shape, Tensorprinter, _bytes, indices
from tensor import Tensor as _Tensor
from tensor import TensorShape, TensorSpec
import math
from random.random import rand as _rand
from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from sys.info import num_physical_cores

fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

@always_inline
fn tensor_op[dtype : DType, func: fn[dtype: DType, nelts: Int] (
        x: SIMD[dtype, nelts], y: SIMD[dtype, nelts]) -> SIMD[dtype, nelts],
](t1: Tensor[dtype], t2: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise operation on two tensors of equal shape."""
    var shape = t1.shape == t2.shape
    var elm = t1.num_elements() == t2.num_elements()
    if shape != elm: 
        print(Error("Both inputs must be the same shape"))
    alias nelts = simdwidthof[dtype]()
    var num_cores = num_physical_cores()
    var res = Tensor[dtype](t1.shape)
    @parameter
    fn calc(i : Int):
        @parameter
        fn vecmath[nelts: Int](idx: Int):
            res.store[nelts](
                idx, func[dtype, nelts](t1.load[nelts](idx), t2.load[nelts](idx))
            )
        vectorize[vecmath, nelts](t1.num_elements())
    parallelize[calc](t1.num_elements(), num_cores)
    return res

@always_inline
fn scalar_ops[dtype : DType, func : fn[type: DType, simd_width: Int](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[type, simd_width]](
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

@always_inline
fn matmul[dtype : DType](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    """Matrix multiplication of two tensors A and B.
    A should be of shape (m, k) and B should be of shape (k, n).
    The result will be a tensor of shape (m, n).
    """
    var m = A.shape[0]  
    var k = A.shape[1]  
    var n = B.shape[1] 
    alias nelts = simdwidthof[dtype]()

    if k != B.shape[0]:
        print("Incompatible shapes for matrix multiplication: A.shape[1] must equal B.shape[0]")
        return A
    
    var result = Tensor[dtype](List[Int](m, n))

    @parameter
    fn multiply_and_sum(i: Int):
        @parameter
        fn index[nelts : Int](j: Int):
            var sum: SIMD[dtype,1] = 0
            for p in range(k):
                sum += A[i, p] * B[p, j]
            result.__setitem__(List[Int](i, j), sum)
        vectorize[index, nelts](n)
    parallelize[multiply_and_sum](m)
    return result

fn matmul_tiled_unrolled_parallelized[dtype : DType](C: Tensor[dtype], A: Tensor[dtype], B: Tensor[dtype]):
    alias nelts = simdwidthof[dtype]()
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.__setitem__(List[Int](m, n + x), C.__getitem__(m, n + x) + A.__getitem__(m, k) * B.__getitem__(k, n + x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, size=tile_x, unroll_factor=unroll_factor]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)

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
        self = scalar_ops[type,math.add](self,x)

    @always_inline
    fn add(inout self: Self, inout other: Tensor[type]) -> Self:
        """
        Performs element-wise addition of two tensors with compatible broadcasted shapes.
        """
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
            var self_indices = result_indices

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    self_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = self.flatten_index(self.shape, self_indices)
            var other_idx = self.flatten_index(other.shape, other_indices)

            result[idx] = math.add(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn sub(inout self : Self, x : SIMD[type,1]):
        self = scalar_ops[type,math.sub](self,x)

    @always_inline
    fn sub(inout self: Self, inout other: Tensor[type]) -> Self:
        """
        Performs element-wise addition of two tensors with compatible broadcasted shapes.
        """
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
            var self_indices = result_indices

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    self_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = self.flatten_index(self.shape, self_indices)
            var other_idx = self.flatten_index(other.shape, other_indices)

            result[idx] = math.sub(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn multiply(inout self : Self, x : SIMD[type,1]):
        self = scalar_ops[type,math.mul](self,x)

    @always_inline
    fn multiply(inout self: Self, inout other: Tensor[type]) -> Self:
        """
        Performs element-wise addition of two tensors with compatible broadcasted shapes.
        """
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
            var self_indices = result_indices

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    self_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = self.flatten_index(self.shape, self_indices)
            var other_idx = self.flatten_index(other.shape, other_indices)

            result[idx] = math.mul(self[self_idx],other[other_idx])
        return result

    @always_inline
    fn div(inout self : Self, x : SIMD[type,1]):
        self = scalar_ops[type,math.div](self,x) 

    @always_inline
    fn div(inout self: Self, inout other: Tensor[type]) -> Self:
        """
        Performs element-wise addition of two tensors with compatible broadcasted shapes.
        """

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
            var self_indices = result_indices

            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    self_indices[j] = 0

            for j in range(other.shape.rank()):
                if other.shape[j] == 1:
                    other_indices[j] = 0

            var self_idx = self.flatten_index(self.shape, self_indices)
            var other_idx = self.flatten_index(other.shape, other_indices)

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
    fn transposed(inout self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:

        var _shape = self.shape._shapelist        
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))

        var tshape = _shape
        
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        
        var ttensor = Tensor[type](tshape)

        @parameter
        fn indexing(index : Int):
            var _indices = indices(_shape, index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            var tindex = ttensor.shape.position(tindices)
            ttensor[tindex] = self[index]
        
        parallelize[indexing](self.num_elements(),num_physical_cores())
        
        return ttensor
        
    @always_inline
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var _shape = self.shape._shapelist        
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))

        var tshape = _shape
        
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        
        var ttensor = Tensor[type](tshape)
        
        @parameter
        fn indexing(index : Int):
            var _indices = indices(_shape, index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            var tindex = ttensor.shape.position(tindices)
            ttensor[tindex] = self[index]
        
        parallelize[indexing](self.num_elements(),num_physical_cores())
        
        self = ttensor

    @always_inline
    fn broadcast(inout self: Self, shapes: shape) -> Self:
        """
        Broadcasts the tensor to a specified shape and returns a new tensor with the broadcasted shape.
        
        Args:
            shapes: The target shape for broadcasting.

        Returns:
            A new Tensor with the broadcasted shape.
        """
        var broadcasted_shape = self.shape.broadcast_shapes(shapes)
        
        if broadcasted_shape.rank() == 0:
            print("Cannot broadcast tensor due to incompatible shapes")

        var result = Tensor[type](broadcasted_shape)

        for idx in range(result.num_elements()):
            var result_indices = indices(broadcasted_shape._shapelist, idx)            
            var original_indices = result_indices
            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    original_indices[j] = 0

            var original_idx = self.flatten_index(self.shape, original_indices)

            result[idx] = self[original_idx]
        return result

    @always_inline
    fn broadcast_to(inout self: Self, shapes: shape):
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
            var original_indices = result_indices
            for j in range(self.shape.rank()):
                if self.shape[j] == 1:
                    original_indices[j] = 0

            var original_idx = self.flatten_index(self.shape, original_indices)

            result[idx] = self[original_idx]
        self = result

    @always_inline
    fn flatten_index(self, shape: shape, indices: List[Int]) -> Int:
        """
        Converts a list of multi-dimensional indices into a flat index based on the provided shape.

        Args:
            shape: The shape of the tensor.
            indices: The list of multi-dimensional indices.

        Returns:
            An integer representing the flat index that corresponds to the given multi-dimensional indices.
        """

        var flat_index = 0
        var stride = 1
        for i in range(shape.rank() - 1, -1, -1):
            flat_index += indices[i] * stride
            stride *= shape[i]
        return flat_index

    @always_inline
    fn reshape(inout self: Self, new_shapes: List[Int]):
        """ Reshape the tensor to the new dimensions."""
        var total = 1
        for i in range(new_shapes.__len__()):
            total *= i
        if not total == self.num_elements():
            print("New shape must contain the same number of elements.")
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