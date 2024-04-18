from tensor import TensorShape, TensorSpec


alias TensorStart = "Tensor("
alias TensorEnd = ")"
alias SquareBracketL = "["
alias SquareBracketR = "]"
alias Truncation = "...,"
alias Strdtype = ", dtype="
alias Strshape = ", shape="
alias Comma = ","
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
        buf += ", "
        buf += str(ptr.load(i))
    return buf


@always_inline
fn _serialize_elements(ptr: DTypePointer, len: Int) -> String:
    var buf : String = String("")

    if len == 0:
        return String("")
    buf += SquareBracketL
    if len < Max_Elem_To_Print:
        buf += complete(ptr, len)
        buf += SquareBracketR
        return buf
    buf += complete(ptr, Max_Elem_Per_Side)
    buf += ", "
    buf += Truncation
    buf += complete(ptr + len - Max_Elem_Per_Side, Max_Elem_Per_Side)
    buf += SquareBracketR
    return buf


@always_inline
fn Tensorprinter[type : DType, print_dtype : Bool = True, print_shape : Bool = True](ptr : DTypePointer[type], shape : shape) -> String:

    var buffer : String = String()
    var rank : Int = shape._rank

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
            
            if (row_elem_count >= Max_Elem_To_Print and row_idx == Max_Elem_Per_Side):
                buffer+="\n"
                buffer+=Truncation
                row_idx = row_elem_count = Max_Elem_Per_Side
        buffer+=SquareBracketR
        matrix_idx+=1

        if(num_matrices >= Max_Elem_To_Print and matrix_idx == Max_Elem_Per_Side):
            buffer+="\n"
            buffer+=Truncation
            matrix_idx = num_matrices = Max_Elem_Per_Side
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
  var shape : Pointer[Int]
  var num_elements : Int
  var _rank : Int
  var _shapelist : List[Int]

  fn __init__(inout self : Self):
    self.shape = Pointer[Int]().alloc(0)
    self.num_elements = 0
    self._rank = 0
    self._shapelist = List[Int]()

  fn __init__(inout self :Self, *dims : Int):
    self.shape = Pointer[Int].alloc(dims.__len__())
    self._shapelist = List[Int]()
    for i in range(dims.__len__()):
      self.shape.store(i, dims[i])
      self._shapelist.append(dims[i])

    self._rank = dims.__len__()
    self.num_elements = num_elements(dims)
    
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

  fn __getitem__(self : Self, inout index : Int) -> Int:
    return self.shape[index if index>=0 else self._rank + index]

  fn __setitem__(self : Self, index : Int, value : Int):
    self.shape[index if index>=0 else self._rank + index] = value
  
  fn __len__(self: Self) -> Int:
    return self._rank
  
  fn __eq__(self : Self, other : Self) -> Bool:
    if self.rank() != other.rank():
      return False
    for i in range(self.rank()):
      if self.shape[i] != other[i]:
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
  
  fn count_elements(self : Self) -> Int:
    return self.num_elements

  fn position(self : Self, indices : List[Int]) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)

  fn position(self : Self, indices : VariadicList[Int]) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)
  
  fn position(self : Self, *indices : Int) -> Int:
    return __get_position(indices, self.rank(), self._shapelist, self.num_elements)


fn __get_position(indices : List[Int], rank : Int, Shapes : List[Int], size : Int) ->Int:
    var pos = 0
    var dim = 1

    for i in range(rank-1,0,-1):
      dim *= Shapes[i]
      pos += (covert_positon(indices[i-1], Shapes[i-1]) * dim)
      pos += (covert_positon(indices[rank-1], Shapes[rank-1]))
    var valid = pos >= 0 and pos < size
    if not valid:
      print("Invalid position")
    return pos


fn __get_position(*indices : Int, rank : Int, Shapes : List[Int], size : Int) ->Int:
    var pos = 0
    var dim = 1

    for i in range(rank-1,0,-1):
      dim *= Shapes[i]
      pos += (covert_positon(indices[i-1], Shapes[i-1]) * dim)
      pos += (covert_positon(indices[rank-1], Shapes[rank-1]))
    var valid = pos >= 0 and pos < size
    if not valid:
      print("Invalid position")
    return pos


fn __get_position(indices : VariadicList[Int], rank : Int, Shapes : List[Int], size : Int) ->Int:
    var pos = 0
    var dim = 1

    for i in range(rank-1,0,-1):
      dim *= Shapes[i]
      pos += (covert_positon(indices[i-1], Shapes[i-1]) * dim)
      pos += (covert_positon(indices[rank-1], Shapes[rank-1]))
    var valid = pos >= 0 and pos < size
    if not valid:
      print("Invalid position")
    return pos


fn covert_positon(index: Int, size: Int) -> Int:
    if index < 0:
        return size + index
    return index


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


fn _bytes(num_elements : Int, type : DType) -> Int:
  """
  Calculates the total number of bytes required to store the elements of an array.

  Args:
      num_elements: The number of elements in the array.
      type : DType of the elements.
  Returns:
      The total number of bytes required to store the elements as an integer.
  """
  var bytes = type.sizeof()

  return (bytes * num_elements)
