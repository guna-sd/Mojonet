from tensor import TensorShape, TensorSpec
import math


alias TensorStart = "Tensor("
alias TensorEnd = ")"
alias SquareBracketL = "["
alias SquareBracketR = "]"
alias Truncation = "...,"
alias Strdtype = ", dtype="
alias Strshape = ", shape="
alias Comma = ", "
alias Max_Elem_To_Print = 7
alias Max_Elem_Per_Side = Max_Elem_To_Print // 2


@always_inline
fn _max(a: Int, b: Int) -> Int:
  """Calculates the maximum of two integers.
  
  Args:
    a: The first integer.
    b: The second integer.
  
  Returns:
    The maximum value between `a` and `b`.
  """
    return a if a > b else b


@always_inline
fn _rank0(type : DType, shape : shape) -> String:
  """
  Generates a string representation for a rank-0 tensor (scalar).
  
  Args:
    type: The data type of the tensor.
    shape: The shape of the tensor.
  
  Returns:
    A string representation of a rank-0 tensor including its type and shape.
  """
    return TensorStart+SquareBracketL+SquareBracketR+Comma+Strdtype+str(type)+Comma+Strshape+shape.__str__()+TensorEnd


@always_inline
fn complete(ptr : DTypePointer, len : Int) -> String:
  """
  Concatenates the elements of a tensor into a string, separated by commas.
  
  Args:
    ptr: A pointer to the data of the tensor elements.
    len: The number of elements to include in the string.
  
  Returns:
    A string representation of the tensor elements.
"""
    var buf = String("")
    if len == 0:
        return buf
    buf += ptr.load()
    for i in range(1, len):
        buf += Comma
        buf += str(ptr.load(i))
    return buf


@always_inline
fn _serialize_elements(ptr: DTypePointer, len: Int) -> String:
  """
  Serializes the elements of a tensor into a string representation, including square brackets.
  
  Args:
    ptr: A pointer to the data type of the tensor elements.
    len: The number of elements to serialize.
  
  Returns:
    A string representation of the tensor elements, enclosed in square brackets.
  """
    var buf = String("")

    if len == 0:
        return String("")
    buf += SquareBracketL
    buf += complete(ptr, len)
    buf += SquareBracketR
    return buf


@always_inline
fn Tensorprinter[type : DType, print_dtype : Bool = True, print_shape : Bool = True](ptr : DTypePointer[type], shape : shape) -> String:
  """Generates a string representation of a tensor, including its data type and shape if specified.

  Parameters:
      type: The data type of the tensor.
      print_dtype: A boolean indicating whether to include the data type in the string representation.
      print_shape: A boolean indicating whether to include the shape in the string representation.
  
  Args:
    ptr: A pointer to the data type of the tensor elements.
    shape: The shape of the tensor.
  
  Returns:
    A string representation of the tensor.
  """
    var buffer = String()
    var rank = shape._rank

    if rank == 0:
        return _rank0(type, shape)
    buffer += TensorStart

    var column_elem_count  = 1 if rank < 1 else shape._shapelist[-1]
    var row_elem_count = 1 if rank < 2 else shape._shapelist[-2]
    
    var matrix_elem_count = column_elem_count * row_elem_count
    
    for i in range(2,rank):
        buffer+=SquareBracketL
    
    var num_matrices = 1

    for i in range(_max(rank -2, 0)):
        num_matrices *= shape._shapelist[i]
    
    var matrix_idx = 0
    while matrix_idx < num_matrices:
        if matrix_idx > 0:
            buffer+=",\n"
        buffer+=SquareBracketL

        var row_idx = 0
        while row_idx < row_elem_count:
            if row_idx > 0:
                buffer+="\n"
            
            buffer += _serialize_elements(
            ptr + matrix_idx * matrix_elem_count + row_idx * column_elem_count,
            column_elem_count,)
            row_idx += 1

            if row_idx != row_elem_count:
                buffer+=","
            
        buffer+=SquareBracketR
        matrix_idx+=1

    for i in range(2,rank):
        buffer+=SquareBracketR
    
    if print_dtype:
        buffer+=Strdtype
        buffer+=type.__str__()
    
    if print_shape:
        buffer+=Strshape
        buffer+=shape.__str__()
    buffer+=TensorEnd
    return buffer


@value
struct shape:
  """Represents the shape of a tensor, encapsulating information about its dimensions."""
  var _ptr : Pointer[Int]
  """_ptr: A pointer to an array of integers, each representing the size of a dimension in the tensor."""
  var num_elements : Int
  """Num_elements: The total number of elements that the tensor can hold, calculated as the product of its dimensions."""
  var _rank : Int
  """_rank: The number of dimensions in the tensor, also known as its rank."""
  var _shapelist : List[Int]
  """_shapelist: A list of integers, each representing the size of a dimension in the tensor, providing an alternative to _ptr for accessing dimension sizes."""

  fn __init__(inout self : Self):
    """Initializes an empty shape."""
    self._ptr = Pointer[Int]().alloc(0)
    self.num_elements = 0
    self._rank = 0
    self._shapelist = List[Int]()

  fn __init__(inout self :Self, *dim : Int):
    """
    Initializes a shape with given dimensions.

    Args:
      dim: A variadic list of integers representing the dimensions of the shape.
    """
    self._ptr = Pointer[Int].alloc(dim.__len__())
    self._shapelist = List[Int]()
    for i in range(dim.__len__()):
      self._ptr.store(i, dim[i])
      self._shapelist.append(dim[i])

    self._rank = dim.__len__()
    self.num_elements = num_elements(dim)
  
  fn __init__(inout self :Self, dim : List[Int]):
    """
    Initializes a shape with given dimensions.
    
    Args:
      dim: A list of integers representing the dimensions of the shape.
    """
    self._ptr = Pointer[Int].alloc(dim.__len__())
    self._shapelist = List[Int]()
    for i in range(dim.__len__()):
      self._ptr.store(i, dim[i])
      self._shapelist.append(dim[i])
    self._rank = dim.__len__()
    self.num_elements = num_elements(dim)
  
  fn __init__[size : Int](inout self :Self, dim : StaticIntTuple[size]):
    """
    Initializes a shape with given dimensions.
    
    Args:
      dim: A tuple of integers representing the dimensions of the shape.
    """
    self._ptr = Pointer[Int].alloc(dim.__len__())
    self._shapelist = List[Int]()
    for i in range(dim.__len__()):
      self._ptr.store(i, dim[i])
      self._shapelist.append(dim[i])

    self._rank = dim.__len__()
    self.num_elements = dim.flattened_length()

  fn __init__(inout self : Self, shape : TensorShape):
    """
    Initializes a shape from a TensorSpec.
    
    Args:
      shape: A TensorShape object.
    """
    self._ptr = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.rank()
    self.num_elements = shape.num_elements()
  
  fn __init__(inout self : Self, shape : TensorSpec):
    """
    Initializes a shape from a TensorSpec.
    
    Args:
      shape: A TensorSpec object.
    """
    self._ptr = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.rank()
    self.num_elements = shape.num_elements()

  fn __copyinit__(inout self: Self, old: Self):
    """Copy initializes a shape from another shape."""
    self._ptr = old._ptr
    self._rank = old._rank
    self.num_elements = old.num_elements
    self._shapelist = old._shapelist
  
  fn __moveinit__(inout self: Self, owned existing: Self):
    """Move initializes a shape, transferring ownership from another shape."""
    self._ptr = existing._ptr
    self._rank = existing._rank
    self.num_elements = existing.num_elements
    self._shapelist = existing._shapelist

  fn __getitem__(self : Self, index : Int) -> Int:
    """Retrieves the dimension size at the given index."""
    return self._ptr[index if index>=0 else self._rank + index]

  fn __setitem__(self : Self, index : Int, value : Int):
    """Sets the size of the dimension at the given index."""
    self._ptr[index if index>=0 else self._rank + index] = value
  
  fn __len__(self: Self) -> Int:
    """Returns the rank (number of dimensions) of the shape."""
    return self._rank
  
  fn __eq__(self : Self, other : Self) -> Bool:
    """Checks if two shapes are equal."""
    if self.rank() != other.rank():
      return False
    for i in range(self.rank()):
      if self._ptr[i] != other[i]:
        return False
    return True
  
  fn __ne__(self : Self, other : Self) -> Bool:
    """Checks if two shapes are not equal."""
    return not self.__eq__(other)
  
  fn __str__(self : Self) -> String:
    """Returns a string representation of the shape."""
    var buf = String("")
    if len(self._shapelist) != 1:
      for i in range(len(self._shapelist)):
        if i:
          buf += "x"
        buf += str(self._shapelist[i])
      return buf
    if self._shapelist.__len__() == 0 and self.num_elements == 0 and self._rank == 0:
      buf+= 'none'
      return buf
    buf += "1x"
    buf += str(self._shapelist[0])
    return buf

  fn rank(self : Self) -> Int:
    """Returns the rank (number of dimensions) of the shape."""
    return self._rank

  @always_inline
  fn broadcast_shapes(inout self, _shape: shape) -> shape:
    """
    Broadcasts two shapes to a common shape.
    
    Args:
      _shape: The shape to broadcast with.
    
    Returns:
      The broadcasted shape.
    """
      var ndim = math.max(self.rank(), _shape.rank())
      var diff = math.abs(self.rank()- _shape.rank())

      var big_shape: shape
      var small_shape: shape
      if self.rank() > _shape.rank():
          big_shape = self
          small_shape = _shape
      else:
          big_shape = _shape
          small_shape = self

      var res = self._shapelist

      for i in range(ndim - 1, diff - 1, -1):
          var a = big_shape[i]
          var b = small_shape[i - diff]
          if b == a:
              res[i] = a
          elif a == 1 or b == 1:
              res[i] = a * b
          else:
              var message: String = "Shapes " + self.__str__() + " and " + str(_shape) + " cannot be broadcasted."
              print(message)
      for i in range(diff - 1, -1, -1):
          res[i] = big_shape[i]

      return shape(res)
  
  fn count_elements(self : Self) -> Int:
    """Returns the total number of elements based on the given shape."""
    return self.num_elements
  
  @always_inline
  fn offset(self : Self, indices : List[Int]) ->Int:
    """Calculates the flat index for a list of multi-dimensional indices.
    
    Args:
      indices: The multi-dimensional indices.
    
    Returns:
      The flat index corresponding to the multi-dimensional indices.
    """
    return flatten_index(self, indices)

  @always_inline
  fn offset(self : Self, indices : VariadicList[Int]) ->Int:
    """Calculates the flat index for a variadic list of multi-dimensional indices.
    
    Args:
      indices: The multi-dimensional indices.
    
    Returns:
      The flat index corresponding to the multi-dimensional indices.
    """
    return flatten_index(self, indices)

  @always_inline
  fn position(self : Self, indices : List[Int]) -> Int:
    """Calculates the flat index for a variadic list of multi-dimensional indices.
    
    Args:
      indices: The multi-dimensional indices.
    
    Returns:
      The flat index corresponding to the multi-dimensional indices.
    """
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)  
  
  @always_inline
  fn indices(self : Self, index : Int) -> List[Int]:
    """
    Converts a linear index into its corresponding multi-dimensional indices based on the given shape. (i.e., its position in a flattened version of the tensor or array).

    Args:
        index: An Int representing the linear index of an element in the flattened tensor or array.
    
    Returns:
        List[Int] containing the multi-dimensional indices corresponding to the given linear index.
    """
    return indices(self._shapelist, index)


@always_inline
fn __get_position(indices : List[Int], rank : Int, Shapes : List[Int], size : Int) ->Int:
    """
    Convert a set of multidimensional indices into a linear index based on the tensor's shape.

    Args:
        indices : (List[Int]) The multidimensional indices to convert.
        rank : (Int) The rank (number of dimensions) of the tensor.
        Shapes : (List[Int]) The shape of the tensor (list of dimensions).
        size : (Int) The total number of elements in the tensor.

    Returns:
        Int: The linear index corresponding to the given multidimensional indices.
    """
    var pos = 0
    var dim = 1
    
    for i in range(rank - 1, -1, -1):
        var index = convert_position(indices[i], Shapes[i])        
        pos += index * dim        
        dim *= Shapes[i]
    if not (0 <= pos < size):
        print(Error("Calculated position is out of bounds."))
    
    return pos


@always_inline
fn convert_position(index: Int, size: Int) -> Int:
    """
    Convert a negative index to a positive one within the given size.

    Args:
        index : The index to convert.
        size : The size of the dimension.

    Returns:
        Int: The positive index within the dimension.
    """
    if index < 0:
        return size + index
    return index


@always_inline
fn indices(shape: List[Int], index: Int) -> List[Int]:
    """
    Converts a linear index into its corresponding multi-dimensional indices based on the given shape.

    This function is useful for determining the multi-dimensional indices of an element in a tensor or array,
    given its linear index (i.e., its position in a flattened version of the tensor or array) and the shape of the tensor or array.

    Args:
        shape: A List[Int] representing the dimensions of the tensor or array.
        index: An Int representing the linear index of an element in the flattened tensor or array.

    Returns:
        A List[Int] containing the multi-dimensional indices corresponding to the given linear index.
    """
    var dim_indices = List[Int]()
    var num_dims = len(shape)
    var linear_index = index
    for i in range(num_dims - 1, -1, -1):
        var dim_size = shape[i]
        var dim_index = linear_index % dim_size
        dim_indices.append(dim_index)       
        linear_index //= dim_size
    dim_indices.reverse()
    return dim_indices


@always_inline
fn flatten_index(shape: shape, indices: List[Int]) -> Int:
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
fn flatten_index(shape: shape, indices: VariadicList[Int]) -> Int:
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
fn num_elements(shape : VariadicList[Int]) -> Int:
    """
    Total number of elements in the given shape.
    
    Args:
      shape: A VariadicList of integers representing the dimensions of the array.

    Returns:
      An integer representing the total number of elements in the array.
    """
    var elements : Int = 1
    for i in range(len(shape)):
        elements *=  shape[i]
    return elements


@always_inline
fn num_elements(shape : List[Int]) -> Int:
    """
    Total number of elements in the given shape.
    
    Args:
      shape: A List of integers representing the dimensions of the array.

    Returns:
      An integer representing the total number of elements in the array.
    """
    var elements : Int = 1
    for i in range(len(shape)):
        elements *=  shape[i]
    return elements


@always_inline
fn _bytes(num_elements : Int, type : DType) -> Int:
  """
  Calculates the total number of bytes required to store the elements of an Tensor.

  Args:
      num_elements: The number of elements in the Tensor.
      type : DType of the elements.
  
  Returns:
      The total number of bytes required to store the elements as an integer.
  """
  var bytes = type.sizeof()

  return (bytes * num_elements)
