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
    return a if a > b else b


@always_inline
fn _rank0(type : DType, shape : shape) -> String:
    return TensorStart+SquareBracketL+SquareBracketR+Comma+Strdtype+str(type)+Comma+Strshape+shape.__str__()+TensorEnd


@always_inline
fn complete(ptr : DTypePointer, len : Int) -> String:
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
    var buf = String("")

    if len == 0:
        return String("")
    buf += SquareBracketL
    buf += complete(ptr, len)
    buf += SquareBracketR
    return buf


@always_inline
fn Tensorprinter[type : DType, print_dtype : Bool = True, print_shape : Bool = True](ptr : DTypePointer[type], shape : shape) -> String:

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
  var _ptr : Pointer[Int]
  var num_elements : Int
  var _rank : Int
  var _shapelist : List[Int]

  fn __init__(inout self : Self):
    self._ptr = Pointer[Int]().alloc(0)
    self.num_elements = 0
    self._rank = 0
    self._shapelist = List[Int]()

  fn __init__(inout self :Self, *dims : Int):
    self._ptr = Pointer[Int].alloc(dims.__len__())
    self._shapelist = List[Int]()
    for i in range(dims.__len__()):
      self._ptr.store(i, dims[i])
      self._shapelist.append(dims[i])

    self._rank = dims.__len__()
    self.num_elements = num_elements(dims)
    
  fn __init__(inout self :Self, shape : VariadicList[Int]):
    self._ptr = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()

    for i in range(shape.__len__()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.__len__()
    self.num_elements = num_elements(shape)
  
  fn __init__(inout self :Self, shape : List[Int]):
    self._ptr = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()
    for i in range(shape.__len__()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])
    self._rank = shape.__len__()
    self.num_elements = num_elements(shape)
  
  fn __init__[size : Int](inout self :Self, shape : StaticIntTuple[size]):
    self._ptr = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()
    for i in range(shape.__len__()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.__len__()
    self.num_elements = shape.flattened_length()

  fn __init__(inout self : Self, shape : TensorShape):
    self._ptr = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.rank()
    self.num_elements = shape.num_elements()
  
  fn __init__(inout self : Self, shape : TensorSpec):
    self._ptr = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self._ptr.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.rank()
    self.num_elements = shape.num_elements()

  fn __copyinit__(inout self: Self, old: Self):
    self._ptr = old._ptr
    self._rank = old._rank
    self.num_elements = old.num_elements
    self._shapelist = old._shapelist
  
  fn __moveinit__(inout self: Self, owned existing: Self):
    self._ptr = existing._ptr
    self._rank = existing._rank
    self.num_elements = existing.num_elements
    self._shapelist = existing._shapelist

  fn __getitem__(self : Self, index : Int) -> Int:
    return self._ptr[index if index>=0 else self._rank + index]

  fn __setitem__(self : Self, index : Int, value : Int):
    self._ptr[index if index>=0 else self._rank + index] = value
  
  fn __len__(self: Self) -> Int:
    return self._rank
  
  fn __eq__(self : Self, other : Self) -> Bool:
    if self.rank() != other.rank():
      return False
    for i in range(self.rank()):
      if self._ptr[i] != other[i]:
        return False
    return True
  
  fn __ne__(self : Self, other : Self) -> Bool:
    return not self.__eq__(other)
  
  fn __str__(self : Self) -> String:
    var buf = String("")
    if len(self._shapelist) != 1:
      for i in range(len(self._shapelist)):
        if i:
          buf += "x"
        buf += str(self._shapelist[i])
      return buf
    buf += "1x"
    buf += str(self._shapelist[0])
    return buf

  fn rank(self : Self) -> Int:
    return self._rank

  @always_inline
  fn broadcast_shapes(inout self, broadcast_shape: shape) -> shape:
      var ndim = math.max(self.rank(), broadcast_shape.rank())
      var diff = math.abs(self.rank() - broadcast_shape.rank())

      var big_shape: shape
      var small_shape: shape
      if self.rank() > broadcast_shape.rank():
          big_shape = self
          small_shape = broadcast_shape
      else:
          big_shape = broadcast_shape
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
              var message: String = "[ERROR] Shapes " + self.__str__() + " and " + str(broadcast_shape) + " cannot be broadcasted."
              print(message)
      for i in range(diff - 1, -1, -1):
          res[i] = big_shape[i]

      return shape(res)
  
  fn count_elements(self : Self) -> Int:
    return self.num_elements
  
  @always_inline
  fn offset(self : Self, indices : List[Int]) ->Int:
    return flatten_index(self, indices)

  @always_inline
  fn offset(self : Self, indices : VariadicList[Int]) ->Int:
    return flatten_index(self, indices)

  @always_inline
  fn position(self : Self, indices : List[Int]) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)

  @always_inline
  fn position(self : Self, indices : VariadicList[Int]) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)
  
  @always_inline
  fn position(self : Self, *indices : Int) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)

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
fn __get_position(*indices : Int, rank : Int, Shapes : List[Int], size : Int) ->Int:
    """
    Convert a set of multidimensional indices into a linear index based on the tensor's shape.

    Args:
        indices : (VariadicList[Int]) The multidimensional indices to convert.
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
fn __get_position(indices : VariadicList[Int], rank : Int, Shapes : List[Int], size : Int) ->Int:
    """
    Convert a set of multidimensional indices into a linear index based on the tensor's shape.

    Args:
        indices : (VariadicList[Int]) The multidimensional indices to convert.
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
fn indices(shape: List[Int], linearindex: Int) -> List[Int]:
    var dim_indices = List[Int]()
    var num_dims = len(shape)
    var linear_index = linearindex
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
fn num_elements[size : Int](shape : StaticIntTuple[size]) -> Int:
    """
    Total number of elements in the given shape.
    
    Args:
      shape: A StaticIntTuple of integers representing the dimensions of the array.

    Returns:
      An integer representing the total number of elements in the array.
    """
    if shape.flattened_length():
      return shape.flattened_length()
    else:
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
