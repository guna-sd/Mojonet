alias TensorStart = "\nTensor("
alias TensorEnd = ")\n"
alias SquareBracketL = "["
alias SquareBracketR = "]"
alias Truncation = "...,"
alias Strdtype = ",  dtype="
alias Strshape = ", shape="
alias Comma = ", "

@value
struct shape:
    """_ptr: A pointer to an array of integers, each representing the size of a dimension in the tensor."""
    var num_elements : Int
    """`num_elements:` The total number of elements that the tensor can hold, calculated as the product of its dimensions."""
    var ndim : Int
    """`ndim:` The number of dimensions in the tensor, also known as its rank."""
    var shapes : List[Int]
    """`shapes:` A list of integers, each representing the size of a dimension in the tensor, providing an alternative to _ptr for accessing dimension sizes."""
    var strides : List[Int]

    @always_inline("nodebug")
    fn __init__(inout self : Self):
      """Initializes an empty shape."""
      self.num_elements = 0
      self.ndim = 0
      self.shapes = List[Int]()
      self.strides = calculate_strides(List[Int]())

    @always_inline("nodebug")
    fn __init__(inout self :Self, *dim : Int):
      """
      Initializes a shape with given dimensions.

      Args:
        dim: A variadic list of integers representing the dimensions of the shape.
      """
      self.shapes = List[Int]()
      for i in range(dim.__len__()):
        self.shapes.append(dim[i])

      self.ndim = dim.__len__()
      self.num_elements = num_elements(list(dim))
      self.strides = calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn __init__(inout self :Self, dim : List[Int]):
      """
      Initializes a shape with given dimensions.
      
      Args:
        dim: A list of integers representing the dimensions of the shape.
      """
      self.shapes = List[Int]()
      for i in range(dim.__len__()):
        self.shapes.append(dim[i])
      self.ndim = dim.__len__()
      self.num_elements = num_elements(dim)
      self.strides = calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn __init__[size : Int](inout self :Self, dim : StaticIntTuple[size]):
      """
      Initializes a shape with given dimensions.
      
      Args:
        dim: A tuple of integers representing the dimensions of the shape.
      """
      self.shapes = List[Int]()
      for i in range(dim.__len__()):
        self.shapes.append(dim[i])

      self.ndim = dim.__len__()
      self.num_elements = dim.flattened_length()
      self.strides = calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn __init__(inout self : Self, shape : TensorShape):
      """
      Initializes a shape from a TensorSpec.
      
      Args:
        shape: A TensorShape object.
      """
      self.shapes = List[Int]()
      for i in range(shape.rank()):
        self.shapes.append(shape[i])

      self.ndim = shape.rank()
      self.num_elements = shape.num_elements()
      self.strides = calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn __init__(inout self : Self, shape : TensorSpec):
      """
      Initializes a shape from a TensorSpec.
      
      Args:
        shape: A TensorSpec object.
      """
      self.shapes = List[Int]()
      for i in range(shape.rank()):
        self.shapes.append(shape[i])

      self.ndim = shape.rank()
      self.num_elements = shape.num_elements()
      self.strides = calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn __copyinit__(inout self: Self, old: Self):
      """Copy initializes a shape from another shape."""
      self.ndim = old.ndim
      self.num_elements = old.num_elements
      self.shapes = old.shapes
      self.strides = old.strides

    @always_inline("nodebug")
    fn __moveinit__(inout self: Self, owned existing: Self):
      """Move initializes a shape, transferring ownership from another shape."""
      self.ndim = existing.ndim
      self.num_elements = existing.num_elements
      self.shapes = existing.shapes
      self.strides = existing.strides

    @always_inline("nodebug")
    fn __getitem__(self : Self, index : Int) -> Int:
      """Retrieves the dimension size at the given index."""
      return self.shapes[index if index>=0 else self.ndim + index]

    @always_inline("nodebug")
    fn __setitem__(inout self : Self, index : Int, value : Int):
      """Sets the size of the dimension at the given index."""
      if index>=0:
        self.shapes.insert(index, value)
      else:
          self.shapes.insert(self.ndim + index, value)

    @always_inline("nodebug")
    fn __len__(self: Self) -> Int:
      """Returns the rank (number of dimensions) of the shape."""
      return self.ndim

    @always_inline("nodebug")
    fn __eq__(self : Self, other : Self) -> Bool:
      """Checks if two shapes are equal."""
      if self.rank() != other.rank():
        return False
      for i in range(self.rank()):
        if self.shapes[i] != other.shapes[i]:
          return False
      return True

    @always_inline("nodebug")
    fn __eq__(self : Self, other : TensorShape) -> Bool:
      """Checks if two shapes are equal."""
      if self.rank() != other.rank():
        return False
      for i in range(self.rank()):
        if self.shapes[i] != other[i]:
          return False
      return True

    @always_inline("nodebug")
    fn __ne__(self : Self, other : Self) -> Bool:
      """Checks if two shapes are not equal."""
      return not self.__eq__(other)

    @always_inline("nodebug")
    fn __str__(self : Self) -> String:
      """Returns a string representation of the shape."""
      var buf = String("")
      if len(self.shapes) != 1:
        for i in range(len(self.shapes)):
          if i:
            buf += "x"
          buf += str(self.shapes[i])
        return buf
      if self.shapes.__len__() == 0 and self.num_elements == 0 and self.ndim == 0:
        buf+= 'none'
        return buf
      buf += "1x"
      buf += str(self.shapes[0])
      return buf

    @always_inline("nodebug")
    fn rank(self : Self) -> Int:
      """Returns the rank (number of dimensions) of the shape."""
      return self.ndim

    @always_inline("nodebug")
    fn Strides(self : Self) -> List[Int]:
      return calculate_strides(self.shapes)

    @always_inline("nodebug")
    fn size(self : Self) -> Int:
      """Returns the total number of elements based on the given shape."""
      return self.num_elements

    @always_inline("nodebug")
    fn offset(self : Self, indices : List[Int]) -> Int:
      """Calculates the flat index for a variadic list of multi-dimensional indices."""
        if indices.__len__() > self.rank():
          print(Error("Number of indices must not exceed tensor dimension"))
          exit(1)
        var offset = 0
        var strides = self.strides
        for i in range(indices.__len__()):
          offset += indices[i] * strides[i]
        return offset

    @always_inline("nodebug")
    fn indices(self : Self, index : Int) -> List[Int]:
      """
      Converts a linear index into its corresponding multi-dimensional indices based on the given shape. (i.e., its position in a flattened version of the tensor or array).

      Args:
          index: An Int representing the linear index of an element in the flattened tensor or array.
      
      Returns:
          List[Int] containing the multi-dimensional indices corresponding to the given linear index.
      """
      return calculate_indices(self.shapes, index)


@always_inline
fn broadcast_shapes(s1: shape, s2: shape) -> shape:
      var shape1 = s1
      var shape2 = s2
      var max_rank = math.max(shape1.__len__(), shape2.__len__())
      var result_shape = List[Int](capacity=max_rank)
      for i in range(max_rank):
          var dim1 = shape1[shape1.__len__() - 1 - i] if i < shape1.__len__() else 1
          var dim2 = shape2[shape2.__len__() - 1 - i] if i < shape2.__len__() else 1
          if dim1 != dim2 and dim1 != 1 and dim2 != 1:
              print("Shapes are not compatible for broadcasting:",str(shape1), " and ",str(shape2))
              exit()
          result_shape.insert(0, math.max(dim1, dim2))
      return shape(result_shape)


@always_inline("nodebug")
fn calculate_strides(shapes : List[Int]) -> List[Int]:
    var stride = 1
    var strides = List[Int](capacity=shapes.__len__())
    for i in range(shapes.__len__() - 1, -1, -1):
        strides[i] = stride
        stride *= shapes[i]
    return strides


@always_inline("nodebug")
fn get_broadcast_index(index: Int, src_shape: shape, result_shape: shape) -> Int:
    var src_index = 0
    var stride = 1
    for i in range(result_shape.ndim):
        var result_dim = result_shape.ndim - 1 - i
        var src_dim = src_shape.ndim - 1 - i
        var result_idx = (index // stride) % result_shape[result_dim]
        var src_idx = result_idx if src_dim >= 0 and src_shape[src_dim] != 1 else 0
        src_index += src_idx * src_shape.strides[src_dim] if src_dim >= 0 else 0
        stride *= result_shape[result_dim]
    return src_index


@always_inline("nodebug")
fn Index(*Shapes : Int) -> StaticIntTuple[8]:
  """Index Function which returns a StaticIntTuple of size 8."""
  return StaticIntTuple[8](Shapes)


@always_inline("nodebug")
fn calculate_indices(shape: List[Int], index: Int) -> List[Int]:
    """
    Converts a linear index into its corresponding multi-dimensional indices based on the given shape.

    This function is useful for determining the multi-dimensional indices of an element in a tensor or array,
    given its linear index (i.e., its position in a flattened version of the tensor or array) and the shape of the tensor or array.

    Args:
        shape: List[Int] representing the dimensions of the tensor or array.
        index: Int representing the linear index of an element in the flattened tensor or array.

    Returns:
        List[Int] containing the multi-dimensional indices corresponding to the given linear index.
    """
    var idx = index
    var indices = List[Int](capacity=shape.size)
    for dim in reversed(shape):
        indices.insert(0, idx % dim[])
        idx //= dim[]
    return indices


@always_inline("nodebug")
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


@always_inline("nodebug")
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


@always_inline("nodebug")
fn num_batches(shape: shape) -> Int:
    """
    Calculates the number of batches in a tensor based on its shape.

    Args:
        shape: The shape of the tensor.

    Returns:
        The number of batches.
    """
    if shape.rank() <= 2:
        return 1
    var num_batches = 1
    for i in range(shape.rank() - 2):
        num_batches *= shape[i]
    return num_batches


@always_inline("nodebug")
fn list(shapes : VariadicList[Int])-> List[Int]:
    var list = List[Int](capacity=shapes.__len__())
    for i in shapes:
        list.append(i)
    return list

@always_inline("nodebug")
fn list[size : Int](shapes : StaticIntTuple[size])-> List[Int]:
    var list = List[Int](capacity=shapes.__len__())
    for i in range(shapes.__len__()):
        list.append(shapes[i])
    return list

@always_inline("nodebug")
fn list(*shapes : Int)-> List[Int]:
    var list = List[Int](capacity=shapes.__len__())
    for i in shapes:
        list.append(i)
    return list

@always_inline("nodebug")
fn fill_list(inout list : List[Int], val : Int)-> List[Int]:
  for i in range(list.__len__()):
    list[i] = val
  return list

@always_inline("nodebug")
fn _bytes[type : DType](num_elements : Int) -> Int:
  """
  Calculates the total number of bytes required to store the elements of an Tensor.

  Parameters:
      type : DType of the elements.

  Args:
      num_elements: The number of elements in the Tensor.

  Returns:
      The total number of bytes required to store the elements as an integer.
  """
  var bytes = sizeof[type]()
  return (bytes * num_elements)


@always_inline("nodebug")
fn round(number : Float64, ndigits : Int)-> Float64:
    """
    Rounds a floating-point number to a specified number of decimal places.
    
    :param number: The number to be rounded.
    :param ndigits: The number of decimal places to round to. If None, round to the nearest integer.
    :return: The rounded number.
    """
    var factor = 10 ** ndigits
    return int(number * factor + 0.5 if number > 0 else number * factor - 0.5) / factor


@always_inline("nodebug")
fn max_elems_row(shape: shape) -> Int:
    """
    Determines the maximum number of elements per row based on the shape of the tensor and the console printing limit.

    Args:
        shape: The shape of the tensor.

    Returns:
        The maximum number of elements per row.
    """
    if shape.num_elements <= 1:
        return shape.num_elements

    var elements_per_row = 1
    for i in range(shape.ndim - 1):
        elements_per_row *= shape.shapes[i]

    return math.min(elements_per_row, 12)

fn tprint[T : Stringable](elem: T) capturing -> None:
  print(elem, end="", flush=True)

@always_inline("nodebug")
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


@always_inline("nodebug")
fn complete(ptr: DTypePointer, len: Int) -> String:
    """
    Concatenates the elements of a tensor into a string, separated by commas, rounded, and formatted based on the specified width.
    
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
        buf += ", "
        buf += ptr.load(i)
    
    return buf


@always_inline("nodebug")
fn _serialize_elements(ptr: DTypePointer, len: Int,max_elements_per_row : Int) -> String:
    """
    Serializes the elements of a tensor into a string representation, including square brackets.

    Args:
        ptr: A pointer to the data type of the tensor elements.
        len: The number of elements to serialize.
        max_elements_per_row: The maximum number of elements to serialize in the row.

    Returns:
        A string representation of the tensor elements, enclosed in square brackets.
    """
    var buf = String("")

    if len == 0:
        return String("")
    buf += SquareBracketL

    var elements_in_row = 0
    for i in range(len):
        if i > 0:
            buf += Comma
        if elements_in_row == max_elements_per_row:
            buf += "\n\t "
            elements_in_row = 0
        buf += complete(ptr + i, 1)
        elements_in_row += 1

    buf += SquareBracketR
    return buf


@always_inline("nodebug")
fn Tensorprinter[printer : fn[T : Stringable](element : T) capturing -> None, type : DType, print_dtype : Bool = False, print_shape : Bool = False](ptr : DTypePointer[type], shape : shape) -> String:
  """Generates a string representation of a tensor, including its data type and shape if specified.

  Parameters:
      printer : A format function to print the tensor data.
      type: The data type of the tensor.
      print_dtype: A boolean indicating whether to include the data type in the string representation.
      print_shape: A boolean indicating whether to include the shape in the string representation.
  
  Args:
    ptr: A pointer to the data type of the tensor elements.
    shape: The shape of the tensor.
  
  Returns:
    A string representation of the tensor.
  """
    var buffer = "\n"
    var rank = shape.ndim

    if rank == 0:
        return _rank0(type, shape)
    printer(TensorStart)

    var column_elem_count  = 1 if rank < 1 else shape.shapes[-1]
    var row_elem_count = 1 if rank < 2 else shape.shapes[-2]
    
    var matrix_elem_count = column_elem_count * row_elem_count
    var max_elements_per_row = max_elems_row(shape)
    
    for i in range(2,rank):
        printer(SquareBracketL)
    
    var num_matrices = 1

    for i in range(math.max(rank -2, 0)):
        num_matrices *= shape.shapes[i]
    
    var matrix_idx = 0
    while matrix_idx < num_matrices:
        if matrix_idx > 0:
            printer(",\n\n\t")
        printer(SquareBracketL)

        var row_idx = 0
        while row_idx < row_elem_count:
            if row_idx > 0:
                printer("\n\t")

            printer(_serialize_elements(
            ptr + matrix_idx * matrix_elem_count + row_idx * column_elem_count,
            column_elem_count,max_elements_per_row))
            row_idx += 1

            if row_idx != row_elem_count:
                printer(", ")
            
        printer(SquareBracketR)
        matrix_idx+=1

    for i in range(2,rank):
        printer(SquareBracketR)
    
    if print_dtype:
        printer(Strdtype)
        printer(type.__str__())
    
    if print_shape:
        printer(Strshape)
        printer(shape.__str__())
    printer(TensorEnd)
    return buffer