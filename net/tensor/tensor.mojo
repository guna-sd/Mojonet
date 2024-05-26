
@value
struct Tensor[type : DType = DType.float32](AnyType, CollectionElement, EqualityComparable, Stringable):
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
    var data : DTypePointer[type]
    """
    The data is a pointer to a block of memory that holds the elements of the tensor.
    """

    fn __init__(inout self):
        self.data = stack_allocation[0,type]()
        self.shape = shape()
        self.dtype = type

    fn __init__(inout self, *shapes : Int,):
        self.shape = shape(shapes)
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.data, self.shape.num_elements)
    
    fn __init__(inout self, shape : shape, data : DTypePointer[type],):
        self.shape = shape
        self.data = DTypePointer[type](self.shape.num_elements)
        self.dtype = type
        memcpy(self.data, data, self.shape.num_elements)

    fn __init__(inout self, *shapes : Int, data : DTypePointer[type],):
        self.shape = shape(shapes)
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memcpy(self.data, data, self.shape.num_elements)
    
    fn __init__(inout self, shape : shape, data : List[Scalar[type]],):
        self.shape = shape
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        if self.shape.num_elements == data.__len__():
            for i in range(self.shape.num_elements):
                self.data[i] = data[i]

    fn __init__(inout self, shape : shape, *data : Scalar[type],):
        self.shape = shape
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        if self.shape.num_elements == data.__len__():
            for i in range(self.shape.num_elements):
                self.data[i] = data[i]

    fn __init__(inout self, shapes : List[Int], *data : Scalar[type],):
        self.shape = shape(shapes)
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        if self.shape.num_elements == data.__len__():
            for i in range(self.shape.num_elements):
                self.data[i] = data[i]

    fn __init__(inout self, shape : shape, data : DTypePointer[type], value : Int,):
        self.shape = shape
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memcpy(self.data, data, self.shape.num_elements)
        self = self.fill(value)

    fn __init__(inout self, shapes : VariadicList[Int],):
        self.shape = shape(shapes)
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.data, self.shape.num_elements)
    
    fn __init__(inout self, shapes : List[Int],):
        self.shape = shape(shapes)
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.data, self.shape.num_elements)
    
    fn __init__[size : Int](inout self, shapes : StaticIntTuple[size],):
        self.shape = shape(shapes)
        self.dtype = type
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        memset_zero(self.data, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : shape,):
        self.shape = shapes
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.data, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : TensorShape,):
        self.shape = shape(shapes)
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.data, self.shape.num_elements)

    fn __init__(inout self : Self, shapes : TensorSpec,):
        self.shape = shape(shapes)
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memset_zero(self.data, self.shape.num_elements)
    
    fn __init__(inout self : Self, data : _Tensor[type],):
        self.shape = shape(data.shape())
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        self.dtype = type
        memcpy(self.data, data._ptr, self.shape.num_elements)

    fn __copyinit__(inout self: Self, other: Self):
        self.shape = other.shape
        self.data = DTypePointer[type]().alloc(self.shape.num_elements)
        memcpy(self.data, other.data, self.shape.num_elements)
        self.dtype = other.dtype
    
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape
        self.data = existing.data
        self.dtype = existing.dtype
    
    @always_inline("nodebug")
    fn load[nelts : Int](self, owned index: Int) -> SIMD[type,nelts]:
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to load.
        
        Args:
            index : The index in the tensor data from which to load the SIMD value. If negative, it is interpreted as an index from the end of the data.

        Returns:
            The SIMD value loaded from the tensor data at the specified index.
        """
        if index < 0:
            index =  self.num_elements() + index
        return self.data.load[width=nelts](index)
    
    @always_inline("nodebug")
    fn store[nelts : Int](self, owned index: Int, value: SIMD[type,nelts]):
        """Loads a SIMD (Single Instruction, Multiple Data) value from the tensor data at the specified index.

        Parameters:
            nelts: The number of elements in the SIMD value to store.
        
        Args:
            index : The index in the tensor data at which to store the SIMD value. If negative, it is interpreted as an index from the end of the data.
            value : The SIMD value to store in the tensor data at the specified index.
        """
        if index < 0:
            index = self.num_elements() + index
        self.data.store[width=nelts](index, value)
    
    @always_inline("nodebug")
    fn load(self, index : Int) -> SIMD[type,1]:
        return self.load[1](index)

    @always_inline("nodebug")
    fn load[nelts : Int](self, *indices : Int) -> SIMD[type,nelts]:
        var pos = self.shape.offset(list(indices))
        return self.load[nelts](pos)

    @always_inline("nodebug")
    fn load(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.offset(list(indices))
        return self.load[1](pos)

    @always_inline("nodebug")
    fn store(self: Self, index: Int, val: SIMD[type, 1]):
        self.store[1](index, val)

    @always_inline("nodebug")
    fn store[nelts : Int](self: Self, *indices: Int, val: SIMD[type, nelts]):
        var pos = self.shape.offset(list(indices))
        self.store[nelts](pos, val)

    @always_inline("nodebug")
    fn store(self: Self, *indices: Int, val: SIMD[type, 1]):
        var pos = self.shape.offset(list(indices))
        self.store(pos, val)

    @always_inline("nodebug")
    fn store(self: Self, indices: List[Int], val: SIMD[type, 1]):
        var pos = self.shape.offset(indices)
        self.store(pos, val)

    fn __getitem__(self: Self, index: Int)-> SIMD[type,1]:
        return self.load[1](index)
    
    fn __getattr__[index : Int](self: Self) -> SIMD[type,1]:
        return self.load[1](index)
    
    fn __getitem__(self, *indices : Int) -> SIMD[type,1]:
        var pos = self.shape.offset(list(indices))
        return self.load[1](pos)

    fn __getitem__[*indices : Int](self) -> SIMD[type,1]:
        var pos = self.shape.offset(list(indices))
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
                if self.data[i] == other.data[i]:
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
        if self.shape == other.shape:
            return tensor_op[type, math.add](self, other)
        else:
            return Broadcast_op[type, math.add](self, other)

    fn __add__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.add](self,other)

    fn __radd__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.add](self,other)

    fn __iadd__(inout self: Self, other: Self):
        if self.shape == other.shape:
            self = tensor_op[type, math.add](self, other)
        else:
            self =  Broadcast_op[type, math.add](self, other)
 
    fn __iadd__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.add](self,other)

    fn __sub__(self: Self, other: Self) -> Self:
        if self.shape == other.shape:
            return tensor_op[type, math.sub](self, other)
        else:
            return Broadcast_op[type, math.sub](self, other)

    fn __sub__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.sub](self,other)

    fn __rsub__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.sub](self,other)

    fn __isub__(inout self: Self, other: Self):
        if self.shape == other.shape:
            self =  tensor_op[type, math.sub](self, other)
        else:
            self =  Broadcast_op[type, math.sub](self, other)
 
    fn __isub__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.sub](self,other)
    
    fn __mul__(self: Self, other: Self) -> Self:
        if self.shape == other.shape:
            return tensor_op[type, math.mul](self, other)
        else:
            return Broadcast_op[type, math.mul](self, other)

    fn __mul__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.mul](self,other)

    fn __rmul__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.mul](self,other)

    fn __imul__(inout self: Self, other: Self):
        if self.shape == other.shape:
            self =  tensor_op[type, math.mul](self, other)
        else:
            self =  Broadcast_op[type, math.mul](self, other)
 
    fn __imul__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.mul](self,other)
    
    fn __truediv__(self: Self, other: Self) -> Self:
        if self.shape == other.shape:
            return tensor_op[type, math.div](self, other)
        else:
            return Broadcast_op[type, math.div](self, other)

    fn __truediv__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.div](self,other)

    fn __rtruediv__(self: Self, other: SIMD[type,1]) -> Self:
        return scalar_op[type,math.div](self,other)

    fn __itruediv__(inout self: Self, other: Self):
        if self.shape == other.shape:
            self =  tensor_op[type, math.div](self, other)
        else:
            self =  Broadcast_op[type, math.div](self, other)
 
    fn __itruediv__(inout self: Self, other: SIMD[type,1]):
        self = scalar_op[type,math.div](self,other)
    
    fn __pow__(self: Self, exponent: Int) -> Self:
        """
        Exponentiation of each element in the tensor by the given exponent.
        """
        var result = self
        @parameter
        fn power[nelts : Int](i :Int):
            result.data[i] = math.pow(self.data[i], exponent)
        vectorize[power,1](self.num_elements())
        return result

    fn __ipow__(inout self: Self, exponent: Int):
        """
        In-place exponentiation of each element in the tensor by the given exponent.
        """
        self = self.__pow__(exponent)

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

    fn __str__[print_dtype : Bool = True, print_shape : Bool = True](self: Self) -> String:
        return Tensorprinter[type, print_dtype, print_shape](self.data, self.shape)
   
    @always_inline("nodebug")
    fn pow(inout self: Self, pow: Int):    
        self = self.__pow__(pow)
    
    @always_inline("nodebug")
    fn add(self : Self, x : SIMD[type,1]) -> Self:
        return self.__add__(x)

    @always_inline("nodebug")
    fn add(self: Self, other: Tensor[type]) -> Self:
        return self.__add__(other)

    @always_inline("nodebug")
    fn sub(self : Self, x : SIMD[type,1]) -> Self:
        return self.__sub__(x)

    @always_inline("nodebug")
    fn sub(self: Self, other: Tensor[type]) -> Self:
        return self.__sub__(other)

    @always_inline("nodebug")
    fn multiply(self : Self, x : SIMD[type,1]) -> Self:
        return self.__mul__(x)

    @always_inline("nodebug")
    fn multiply(self: Self, other: Tensor[type]) -> Self:
        return self.__mul__(other)

    @always_inline("nodebug")
    fn div(self : Self, x : SIMD[type,1]) -> Self:
        return self.__truediv__(x)

    fn div(self: Self, other: Tensor[type]) -> Self:
        return self.__truediv__(other)
    
    @always_inline("nodebug")
    fn sum(self: Self) -> Scalar[type]:
        var result = Scalar[type]()
        alias nelts = 1#simdwidthof[type]()
        @parameter
        fn _sum[nelts : Int](i : Int):
            result += self[i].reduce_add()
        vectorize[_sum,nelts](self.num_elements())
        return result

    @always_inline("nodebug")
    fn max(self: Self) -> Scalar[type]:
        """
        Find the maximum value in the tensor.

        Returns:
            The maximum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = 1#simdwidthof[type]()
        @parameter
        fn _max[nelts : Int](i : Int):
            result = math.max(result, self[i])
        vectorize[_max,nelts](self.num_elements())
        return result

    @always_inline("nodebug")
    fn min(self: Self) -> Scalar[type]:
        """
        Find the minimum value in the tensor.

        Returns:
            The minimum value in the tensor as a scalar value.
        """
        var result = Scalar[type]()
        alias nelts = 1#simdwidthof[type]()
        @parameter
        fn _min[nelts : Int](i : Int):
            result = math.min(result, self[i])
        vectorize[_min,nelts](self.num_elements())
        return result

    @always_inline("nodebug")
    fn mean(self: Self) -> Scalar[type]:
        """
        Compute the mean (average) value of the tensor.

        Returns:
            The mean value of the tensor as a scalar value.
        """
        return self.sum() / Scalar[type](self.num_elements())

    @always_inline("nodebug")
    fn prod(self: Self) -> Scalar[type]:
        """
        Compute the product of all elements in the tensor.

        Returns:
            The product of all elements in the tensor as a scalar value.
        """
        var result = Scalar[type](1)
        alias nelts = 1#simdwidthof[type]()
        @parameter
        fn _prod[nelts : Int](i : Int):
            result *= self[i].reduce_mul()
        vectorize[_prod,nelts](self.num_elements())
        return result
    
    @always_inline("nodebug")
    fn list(self) -> List[Scalar[type]]:
        var result = List[Scalar[type]]()
        @parameter
        fn lis[nelts : Int](i : Int):
            result.append(self.data.load(i))
        vectorize[lis,1](self.num_elements())
        return result

    @always_inline("nodebug")
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
        @parameter
        fn arng[nelts : Int](i : Int):
            result[i] = value
            value += step
        vectorize[arng,1](self.num_elements())
        return result
    
    @always_inline("nodebug")
    fn arange(inout self):
        self = self.arange(0,self.num_elements())

    @always_inline("nodebug")
    fn zeros(self : Self):
        memset_zero(self.data, self.shape.num_elements)
    
    @always_inline("nodebug")
    fn ones(self : Self):
        memset[type](self.data, 1, self.num_elements())

    @always_inline("nodebug")
    fn rand(self, seed : Optional[Int] = None):
        rfill[type](self.data, self.num_elements())

    @always_inline("nodebug")
    fn random(self, seed : Optional[Int] = None) -> Self:
        rfill(self.data, self.num_elements())
        return self

    @always_inline("nodebug")
    fn fill(self: Self, value: Scalar[type]) -> Self:
        """Fill the tensor with a specified value."""
        var result = DTypePointer[type]().alloc(self.num_elements())
        alias nelts = 1#simdwidthof[type]()

        @parameter
        fn _set[nelts : Int](index: Int):
            self.store[nelts](index, self.load[nelts](index).splat(value))

        vectorize[_set, nelts](self.num_elements())
        return Self(self.shape, result)

    @always_inline("nodebug")
    fn ifill(inout self: Self, value: Scalar[type]):
        """Fill the tensor with a specified value."""
        self = self.fill(value)

    @always_inline("nodebug")
    fn transposed(self: Self, dim1: Int = -2, dim2: Int = 1) -> Self:
        
        if dim1 >= self.rank() or dim2 >= self.rank():
            print(Error("dim1 and dim2 must be within the range of the tensor's rank"))
            abort(external_call["exit", Int](1))

        var tshape = self.shape
        tshape[dim1], tshape[dim2] = tshape[dim2], tshape[dim1]
        var ttensor = Tensor[type](tshape)
        
        for index in range(self.num_elements()):
            var _indices = self.shape.indices(index)
            var tindices = _indices
            tindices[dim1], tindices[dim2] = _indices[dim2], _indices[dim1]            
            ttensor[tindices] = self[index]
        
        return ttensor
        
    @always_inline("nodebug")
    fn transpose(inout self: Self, dim1: Int = -2, dim2: Int = 1):
        var ttensor = self.transposed(dim1, dim2)
        self = Self(ttensor.shape, ttensor.data)

    @always_inline("nodebug")
    fn broadcast(self: Self, shapes: shape) -> Self:
        """
        Broadcasts the tensor to a specified shape and returns a new tensor with the broadcasted shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        return Broadcast_op[type, math.add](self, Tensor[type](shapes))

    @always_inline("nodebug")
    fn broadcast_to(inout self: Self, shapes: shape):
        """
        Broadcasts the tensor to a specified shape.
        
        Args:
            shapes: The target shape for broadcasting.
        """
        var result = self.broadcast(shapes)
        self = Self(result.shape, result.data)

    @always_inline("nodebug")
    fn reshape(self: Self, other: shape) -> Self:
        """ Reshape the tensor to the new dimensions and returns the reshaped tensor."""
        if self.shape.num_elements != other.num_elements:
            print("Error: Cannot reshape tensor.")
            abort(external_call["exit", Int](1))

        var data = Tensor[type](other)
        @parameter
        fn _reshape[nelts : Int](idx : Int):
            var old_indices = self.shape.indices(idx)
            var new_indices = other.indices(idx)
            data[new_indices] = self[old_indices]
        vectorize[_reshape,1](self.num_elements())
        return Self(other, data.data)

    @always_inline("nodebug")
    fn reshape(self: Self, *new: Int) -> Self:
        """
        Reshape the tensor to the specified new shape.
        
        Args:
            new: The new shape dimensions as variadic integers.

        Returns:
            A new tensor reshaped to the new dimensions specified.
        """   
        return self.reshape(shape(new))

    @always_inline("nodebug")
    fn reshape_to(inout self: Self, other: shape):
        """ Reshape the tensor to the new dimensions."""
        self = Self(other, self.reshape(other).data)

    @always_inline("nodebug")
    fn flatten(self: Self) -> Self:
        """
        Flatten the tensor into a 1D array.
        
        Returns:
            A new tensor with all elements in a single dimension.
        """
        var new_shape = shape(self.shape.num_elements)
        return self.reshape(new_shape)

    @always_inline("nodebug")
    fn rank(self: Self) -> Int:
        return self.shape.ndim

    @always_inline("nodebug")
    fn shapes(self: Self) ->shape:
        return self.shape
        
    @always_inline("nodebug")
    fn _dtype(self: Self) -> String:
        return self.dtype.__str__()
    
    @always_inline("nodebug")
    fn num_elements(self: Self) -> Int:
        return self.shape.num_elements
    
    @always_inline("nodebug")
    fn astype[dtype : DType](self : Self) -> Tensor[dtype]:
        var casted = Tensor[dtype](self.shape)
        alias nelts = 1#simdwidthof[dtype]()
        @parameter
        fn cast_single_element[nelts : Int](index: Int):
            casted.store[nelts](index, self[index].cast[dtype]())

        vectorize[cast_single_element, nelts](self.num_elements())
        return casted

    @always_inline("nodebug")
    fn num_bytes(self: Self) -> Int:
        return sizeof[type]() * self.shape.num_elements
    
    @always_inline("nodebug") 
    fn itemsize(self: Self) -> Int:
        return sizeof[type]()

    @always_inline("nodebug")
    fn serialize(self) -> List[Bytes]:
        var bytes = List[Bytes](capacity=self.num_elements())
        for i in range(self.num_elements()):
            bytes.append(tobytes[type](self.data[i]))
        return bytes

    fn deserialize(self, data: List[Bytes], shape: List[Int]) -> Tensor[type]:
        var num_elements = num_elements(shape)
        var tensor_data = List[Scalar[type]](capacity=num_elements)
        for i in range(num_elements):
            var value = frombytes[type](data[i])
            tensor_data.append(value)
        return Tensor[type](shape, tensor_data)
    

    @always_inline("nodebug")
    fn saver(self: Self, path : String):
        alias makgic = 2
        with fopen(path, 'wb') as f:
            f.write("mnetbfile\n".__str__())
            f.write("!s!")
            f.write(__type_of(self.shape.shapes).__str__(self.shape.shapes))
            f.write("\n")
            f.write("!T!")
            f.write(__type_of(self.list()).__str__(self.list()))
            f.write("\n")

fn tensor[dtype : DType = DType.float32](Shape : List[Int], rand : Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor

fn tensor[dtype : DType = DType.float32](*Shape : Int, rand : Bool = False) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if rand:
        tensor.rand()
        return tensor
    tensor.zeros()
    return tensor

fn ones[dtype : DType = DType.float32](Shape : List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](1))
    return tensor

fn ones[dtype : DType = DType.float32](*Shape : Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](1))
    return tensor

fn zeros[dtype : DType = DType.float32](Shape : List[Int],) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](0))
    return tensor

fn zeros[dtype : DType = DType.float32](*Shape : Int,) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](0))
    return tensor

@always_inline
fn fill[dtype : DType = DType.float32](shape : shape, val: Scalar[dtype]) -> Tensor[dtype]:
    var result = Tensor[dtype](shape)
    alias nelts = simdwidthof[dtype]()
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        result.store[nelts](idx, result.load[nelts](idx).splat(val))

    vectorize[fill_vec, nelts](result.num_elements())
    return result

@always_inline
fn fill[dtype : DType = DType.float32](inout x: Tensor[dtype], val: Scalar[dtype]):
    alias nelts = simdwidthof[dtype]()
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        x.store[nelts](idx, x.load[nelts](idx).splat(val))

    vectorize[fill_vec, nelts](x.num_elements())

fn fill[dtype : DType = DType.float32](*Shape : Int, val: Scalar[dtype]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](val))
    return tensor

fn fill[dtype : DType = DType.float32](Shape : List[Int], val: Scalar[dtype]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    fill[dtype](tensor,Scalar[dtype](val))
    return tensor

fn rand[dtype : DType = DType.float32](Shape : List[Int], seed : Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand(seed)
    tensor.rand()
    return tensor

fn rand[dtype : DType = DType.float32](*Shape : Int, seed : Optional[Int]) -> Tensor[dtype]:
    var tensor = Tensor[dtype](Shape)
    if seed:
        tensor.rand(seed)
    tensor.rand()
    return tensor