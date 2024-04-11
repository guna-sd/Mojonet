from tensor import TensorShape, TensorSpec
from tensor import Tensor as _Tensor

@value
struct shape:
  var shape : Pointer[Int]
  var num_elements : Int
  var _rank : Int
  var _shapelist : List[Int]

  fn __init__(inout self : Self):
    self.shape = Pointer[Int]().alloc(0)
    self.num_elements = 0
    self._rank = 0
    self._shapelist = List[Int]()

  fn __init__(inout self :Self, shape : VariadicList[Int]):
    self.shape = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()

    for i in range(shape.__len__()):
      self.shape.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.__len__()
    self.num_elements = num_elements(shape)

  
  fn __init__(inout self :Self, shape : List[Int]):
    self.shape = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()
    for i in range(shape.__len__()):
      self.shape.store(i, shape[i])
      self._shapelist.append(shape[i])
    self._rank = shape.__len__()
    self.num_elements = num_elements(shape)
  
  fn __init__[size : Int](inout self :Self, shape : StaticIntTuple[size]):
    self.shape = Pointer[Int].alloc(shape.__len__())
    self._shapelist = List[Int]()
    for i in range(shape.__len__()):
      self.shape.store(i, shape[i])
      self._shapelist.append(shape[i])

    self._rank = shape.__len__()
    self.num_elements = shape.flattened_length()

  fn __init__(inout self :Self, *dims : Int):
    self.shape = Pointer[Int].alloc(dims.__len__())
    self._shapelist = List[Int]()
    for i in range(dims.__len__()):
      self.shape.store(i, dims[i])
      self._shapelist.append(dims[i])
    self._rank = dims.__len__()
    self.num_elements = num_elements(dims)

  fn __init__(inout self : Self, shape : TensorShape):
    self.shape = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self.shape.store(i, shape[i])
      self._shapelist.append(shape[i])
    self._rank = shape.rank()
    self.num_elements = shape.num_elements()
  
  fn __init__(inout self : Self, shape : TensorSpec):
    self.shape = Pointer[Int].alloc(shape.rank())
    self._shapelist = List[Int]()
    for i in range(shape.rank()):
      self.shape.store(i, shape[i])
      self._shapelist.append(shape[i])
    self._rank = shape.rank()
    self.num_elements = shape.num_elements()
  
  fn __init__(inout self : Self, data : _Tensor):
    self.shape = Pointer[Int].alloc(data.rank())
    self._shapelist = List[Int]()
    for i in range(data.rank()):
      self._shapelist.append(data.shape().__getitem__(i))
    self._rank = data.rank()
    self.num_elements = data.num_elements()

  fn __copyinit__(inout self: Self, old: Self):
    self.shape = old.shape
    self._rank = old._rank
    self.num_elements = old.num_elements
    self._shapelist = old._shapelist
  
  fn __moveinit__(inout self: Self, owned existing: Self):
    self.shape = existing.shape
    self._rank = existing._rank
    self.num_elements = existing.num_elements
    self._shapelist = existing._shapelist

  fn __getitem__(self : Self, index : Int) -> Int:
    return self.shape[index if index>=0 else self._rank + index]

  fn __setitem__(self : Self, index : Int, value : Int):
    self.shape[index if index>=0 else self._rank + index] = value
  
  fn rank(self : Self) -> Int:
    return self._rank
  
  fn count_elements(self : Self) -> Int:
    return self.num_elements
  
  fn __eq__(self : Self, other : TensorShape) -> Bool:
    if self.rank() != other.rank():
      return False
    if self.num_elements != other.num_elements():
      return False
    for i in range(self.rank()):
      if self.shape[i] != other[i]:
        return False
    return True
  
  fn __ne__(self : Self, other : TensorShape) -> Bool:
    return not self.__eq__(other)
  
  fn __str__(self : Self) -> String:
    var buf = String("")
    for i in range(len(self._shapelist)):
      if i:
        buf += "x"
      buf += str(self._shapelist[i])
    return buf

fn num_elements(shape : VariadicList[Int]) -> Int:
    """
    Total number of elements in the given shape.
    
    Args:
      shape: A VariadicList of integers representing the dimensions of the array.

    Returns:
      An integer representing the total number of elements in the array.
    """
    var elements : Int = 1
    var dim : Int
    for i in range(len(shape)):
        dim = shape[i]
        elements = elements * dim
    return elements

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
      var dim : Int
      for i in range(len(shape)):
          dim = shape[i]
          elements = elements * dim
      return elements

fn num_elements(shape : List[Int]) -> Int:
    """
    Total number of elements in the given shape.
    
    Args:
      shape: A List of integers representing the dimensions of the array.

    Returns:
      An integer representing the total number of elements in the array.
    """
    var elements : Int = 1
    var dim : Int
    for i in range(len(shape)):
        dim = shape[i]
        elements = elements * dim
    return elements

fn calculate_strides(shape: List[Int], type : DType) -> List[Int]:
  """
  Calculates the strides for each dimension of an array given its shape and element size.

  Args:
      shape: A list of integers representing the dimensions of the array.
      type : DType of the array elements.

  Returns:
      A list of integers representing the strides for each dimension.
  """

  var strides : List[Int] = List[Int]()
  var typesize : Int = type.sizeof()
  var p = 1

  for i in range(len(shape)-1,-1,-1):
    strides.append(p)
    p = p * shape[i]
  strides.reverse()
  for i in range(len(strides)):
    strides[i] = strides[i] * typesize

  return strides

fn _bytes(num_elements : Int, type : DType) -> Int:
  """
  Calculates the total number of bytes required to store the elements of an array.

  Args:
      num_elements: The number of elements in the array.
      type : DType of the array elements.
  Returns:
      The total number of bytes required to store the elements as an integer.
  """
  var bytes = type.sizeof()

  return (bytes * num_elements)
