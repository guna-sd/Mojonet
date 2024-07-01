from builtin.io import _snprintf_scalar
from builtin.string import _calc_format_buffer_size
from utils import StaticIntTuple
from utils._format import Formatter

alias TensorStart = "Tensor("
alias TensorEnd = ")"
alias SquareBracketL = "["
alias SquareBracketR = "]"
alias Truncation = " ...,"
alias CompactMaxElemsToPrint = 19
alias CompactElemPerSide = 4

@register_passable("trivial")
struct shape:
    alias maxrank = 26
    alias shape_type = StaticIntTuple[Self.maxrank]

    var shapes : Self.shape_type
    """`shapes:` shapes of ndimensional tensors."""
    var strides : Self.shape_type
    """`strides:` strides between ndimensional tensors."""
    var rank : Int
    """`rank:` The number of dimensions in the tensor, also known as its rank."""
    var num_elements : Int
    """`num_elements:` The total number of elements that a tensor can hold based on the shape."""

    @always_inline("nodebug")
    fn __init__(inout self):
        self.shapes = StaticIntTuple[self.maxrank]()
        self.strides = calculate_strides(self.shapes)
        self.num_elements = num_elements(list(self.shapes, 0))
        self.rank = 0

    @always_inline("nodebug")
    fn __init__[*element_types : Intable](inout self, owned *elements : *element_types):
        if (len(elements) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = StaticIntTuple[Self.maxrank]()

        @parameter
        for i in range(0, elements.__len__()):
            self.shapes[i] = int(elements.__getitem__[i]())
        
        self.strides = calculate_strides(self.shapes)
        self.num_elements = num_elements(list(self.shapes, len(elements)))
        self.rank = len(elements)

    @always_inline("nodebug")
    fn __init__(inout self, owned shapes : List[Int]):
        if (len(shapes) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = StaticIntTuple[self.maxrank]()
        for i in range(len(shapes)):
            self.shapes[i] = shapes[i]
        self.strides = calculate_strides(self.shapes)
        self.num_elements = num_elements(list(self.shapes, len(shapes)))
        self.rank = len(shapes)

    @always_inline("nodebug")
    fn __init__(inout self, owned shapes : VariadicList[Int]):
        if (len(shapes) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = StaticIntTuple[self.maxrank]()
        for i in range(len(shapes)):
            self.shapes[i] = shapes[i]
        self.strides = calculate_strides(self.shapes)
        self.num_elements = num_elements(list(self.shapes, len(shapes)))
        self.rank = len(shapes)
    
    @always_inline("nodebug")
    fn __init__(inout self, owned shapes : TensorSpec):
        if (shapes.rank() > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = StaticIntTuple[self.maxrank]()
        for i in range(shapes.rank()):
            self.shapes[i] = shapes[i]
        self.strides = calculate_strides(self.shapes)
        self.num_elements = shapes.num_elements()
        self.rank = shapes.rank()
    
    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __eq__(self : Self, other : TensorShape) -> Bool:
      if self.rank != other.rank():
        return False
      for i in range(self.rank):
        if self.shapes[i] != other[i]:
          return False
      return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __ne__(self : Self, other : TensorShape) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.rank
    
    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        if index > self.rank:
            print("index out of bounds")
            exit(1)
        return self.shapes[index if index >= 0 else self.rank + index]

    @always_inline("nodebug")
    fn __setitem__(inout self, index: Int, value: Int):
        if index > self.rank:
            print("index out of bounds")
            exit(1)
        self.shapes[index if index >= 0 else self.rank + index] = value

    @always_inline("nodebug")
    fn __repr__(self : Self) -> String:
        var buf = String("")
        if len(self) != 1:
            for i in range(len(self)):
                if i:
                    buf += "x"
                buf += str(self.shapes[i])
            return buf
        if self.rank == 0:
            buf+= 'none'
            return buf
        buf += "1x"
        buf += str(self.shapes[0])
        return buf^

    @always_inline("nodebug")
    fn __str__(self : Self) -> String:
        var buffer: String = "("
        for i in range(len(self)):
            buffer+=str(self[i])
            if i != len(self) - 1:
                buffer+=", "
        buffer += ")"
        return buffer

    @always_inline("nodebug")
    fn Strides(self : Self) -> List[Int]:
        var list = List[Int](capacity=self.rank)
        for i in range(self.rank):
            list.append(self.shapes[i])
        return list^

    @always_inline("nodebug")
    fn Shapes(self) -> List[Int]:
        var list = List[Int](capacity=self.rank)
        for i in range(self.rank):
            list.append(self.shapes[i])
        return list^
    
    @always_inline("nodebug")
    fn Rank(self : Self) -> Int:
      """Returns the rank (number of dimensions)."""
        return self.rank
    
    @always_inline("nodebug")
    fn numofelements(self : Self) -> Int:
        return self.num_elements
    
    @always_inline("nodebug")
    fn reduce(self) -> Int:
        var ret : Int = 1
        for i in (self.Shapes()):
            ret *= i[]
        return ret

    @always_inline("nodebug")
    fn offset(self : Self, *indices : Int) -> Int:
      """Calculates the flat index for a variadic list of multi-dimensional indices."""
        if indices.__len__() > self.__len__():
          print("Number of indices must not exceed tensor dimension")
          exit(1)
        var offset = 0
        var strides = self.strides
        for i in range(indices.__len__()):
          offset += indices[i] * strides[i]
        return offset

    @always_inline("nodebug")
    fn offset(self : Self, indices : List[Int]) -> Int:
      """Calculates the flat index for a variadic list of multi-dimensional indices."""
        if indices.__len__() > self.__len__():
          print("Number of indices must not exceed tensor dimension")
          exit(1)
        var offset = 0
        var strides = self.strides
        for i in range(indices.__len__()):
          offset += indices[i] * strides[i]
        return offset
    
    @always_inline("nodebug")
    fn indices(self : Self, idx : Int) -> List[Int]:
        """Converts a linear index into its corresponding multi-dimensional indices based on the shape."""
        return calculate_indices(self.shapes, idx)


@always_inline("nodebug")
fn calculate_strides[size : Int](shapes : StaticIntTuple[size]) -> StaticIntTuple[size]:
    var stride = 1
    var strides = StaticIntTuple[size]()
    for i in range(shapes.__len__() - 1, -1, -1):
        strides[i] = stride
        stride *= shapes[i]
    return strides


@always_inline("nodebug")
fn calculate_indices[size: Int](shape: StaticIntTuple[size], index: Int) -> List[Int]:
    var idx = index
    var indices = List[Int](capacity=size)
    @parameter
    for i in reversed(range(0,size)):
        indices[i] = idx % shape[i]
        idx //= shape[i]
    return indices^


@always_inline("nodebug")
fn possible_cross_dimension_overlap(sizes: List[Int], strides: List[Int]) -> Bool:
    var n_dim = len(sizes)
    var stride_indices = List[Int](capacity=n_dim)
    for i in range(n_dim):
        stride_indices.append(i)
    
    for i in range(1, n_dim):
        var c = i
        for j in range(i - 1, -1, -1):
            if strides[stride_indices[j]] > strides[stride_indices[c]]:
                stride_indices[j], stride_indices[c] = stride_indices[c], stride_indices[j]
                c = j
    for i in range(1, n_dim):
        if sizes[stride_indices[i]] != 1 and strides[stride_indices[i]] < sizes[stride_indices[i-1]] * strides[stride_indices[i-1]]:
            return True
    
    return False


@always_inline("nodebug")
fn are_expandable(shape1: shape, shape2: shape) -> Bool:
    var ndim1 = shape1.rank
    var ndim2 = shape2.rank
    var ndim = min(ndim1, ndim2)

    for _ in range(ndim - 1, -1, -1):
        ndim1 -= 1
        ndim2 -= 1
        if shape1[ndim1] == shape2[ndim2] or shape1[ndim1] == 1 or shape2[ndim2] == 1:
            continue
        return False
    return True

@always_inline("nodebug")
fn is_compatible(A: List[Int], B: List[Int]) -> Bool:
    for i in range(len(A)):
        if A[i] != B[i]:
            print(
                "Incompatible Shapes: Tensors must have the same shape got [",
                A[i],
                "] and [",
                B[i],
                "] at [",
                i,
                "]",
            )
            return False
    return True

fn handle_issue(msg: String):
    print("Issue: " + msg)
    exit(1)

@always_inline("nodebug")
fn broadcast_shapes(shape1: shape, shape2: shape) -> shape:
      var max_rank = max(shape1.__len__(), shape2.__len__())
      var result_shape = List[Int](capacity=max_rank)
      for i in range(max_rank):
          var dim1 = shape1[shape1.__len__() - 1 - i] if i < shape1.__len__() else 1
          var dim2 = shape2[shape2.__len__() - 1 - i] if i < shape2.__len__() else 1
          if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            handle_issue("Shapes are not compatible for broadcasting:" + str(shape1) + " and " +str(shape2))
          result_shape.insert(0, max(dim1, dim2))
      return shape(result_shape)


@always_inline("nodebug")
fn get_broadcast_index(index: Int, src_shape: shape, result_shape: shape) -> Int:
    var src_index = 0
    var stride = 1
    for i in range(result_shape.rank):
        var result_dim = result_shape.rank - 1 - i
        var src_dim = src_shape.rank - 1 - i
        var result_idx = (index // stride) % result_shape[result_dim]
        var src_idx = result_idx if src_dim >= 0 and src_shape[src_dim] != 1 else 0
        src_index += src_idx * src_shape.Strides()[src_dim] if src_dim >= 0 else 0
        stride *= result_shape[result_dim]
    return src_index


@always_inline("nodebug")
fn calculate_indices(shape: List[Int], index: Int) -> List[Int]:
    var idx = index
    var indices = List[Int](capacity=shape.size)
    for dim in reversed(shape):
        indices.insert(0, idx % dim[])
        idx //= dim[]
    return indices^


@always_inline("nodebug")
fn flatten_index(shape: shape, indices: List[Int]) -> Int:
    var flat_index = 0
    var stride = 1
    for i in range(shape.rank - 1, -1, -1):
        flat_index += indices[i] * stride
        stride *= shape[i]
    return flat_index


@always_inline("nodebug")
fn num_batches(shape: shape) -> Int:
    if shape.rank <= 2:
        return 1
    var num_batches = 1
    for i in range(shape.rank - 2):
        num_batches *= shape[i]
    return num_batches


@always_inline("nodebug")
fn num_elements(shape : List[Int]) -> Int:
    var elements : Int = 1
    for i in range(len(shape)):
        elements *=  shape[i]
    return elements


@always_inline("nodebug")
fn list(shapes : VariadicList[Int])-> List[Int]:
    var list = List[Int](capacity=shapes.__len__())
    for i in shapes:
        list.append(i)
    return list^

@always_inline("nodebug")
fn list[size : Int](shapes : StaticIntTuple[size], capacity : Int)-> List[Int]:
    var list = List[Int](capacity=capacity)
    for i in range(capacity):
        list.append(shapes[i])
    return list^

@always_inline("nodebug")
fn list(*shapes : Int)-> List[Int]:
    var list = List[Int](capacity=shapes.__len__())
    for i in shapes:
        list.append(i)
    return list^

@always_inline("nodebug")
fn bytes[type : DType](num_elements : Int) -> Int:
  var bytes = sizeof[type]()
  return (bytes * num_elements)

# ===-------------------------------------------------------------------------------=== #
# Utilities for Serialising the elements of a tensor into a string representation
# ===-------------------------------------------------------------------------------=== #

@always_inline("nodebug")
fn complete[type : DType,](ptr: DTypePointer, len: Int, inout writer : Formatter):
    """
    Concatenates the elements of a tensor into a string, separated by commas, rounded, and formatted based on the specified width."""
    if len == 0:
        return
    _format_scalar[type](writer, rebind[Scalar[type]](ptr.load[width=1]()))
    for i in range(1,len):
        writer.write_str(", ")
        _format_scalar[type](writer, rebind[Scalar[type]](ptr.load[width=1](i)))


@always_inline("nodebug")
fn _serialize_elements[type : DType,](ptr: DTypePointer, len: Int, inout writer : Formatter):
    """
    Serializes the elements of a tensor into a string representation, including square brackets.
    """
    writer.write_str(SquareBracketL)
    var maxelemstoprint: Int = CompactMaxElemsToPrint
    if type.is_int32() or type.is_uint32():
        maxelemstoprint = 15
    if type.is_int64() or type.is_uint64():
        maxelemstoprint = 9

    var maxelemsperside: Int = CompactElemPerSide
    if type.is_int64() or type.is_uint64():
        maxelemsperside = 3
    if len == 0:
        writer.write_str(SquareBracketR)
        return

    if len < maxelemstoprint:
        complete[type](ptr, len, writer)
        writer.write_str(SquareBracketR)
        return

    complete[type](ptr, maxelemsperside, writer)
    writer.write_str(", ")
    writer.write_str(Truncation)
    complete[type](ptr + len - maxelemsperside, maxelemsperside, writer)

    writer.write_str(SquareBracketR)
    return


@always_inline("nodebug")
fn _format_scalar[
    dtype: DType,
    float_format: StringLiteral = "%.4f",
](inout writer: Formatter, value: Scalar[dtype]):
    """
    Formats a scalar value and writes it to the provided formatter.
    """
    alias size: Int = _calc_format_buffer_size[dtype]()
    var buf = InlineArray[UInt8, size](fill=0)

    if dtype.is_floating_point():
        var wrote = _snprintf_scalar[dtype, float_format](
            buf.unsafe_ptr(),
            size,
            value,
        )
        var str_slice = StringSlice[False, __lifetime_of(buf)](
            unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=wrote
        )
        writer.write_str(str_slice)
    else:
        var max_width = len(str(Scalar[dtype].MAX_FINITE))
        var pad = str(" ")
        var type_width = int(dtype.sizeof()/2) -1

        if dtype.is_int64() or dtype.is_uint64():
            type_width = type_width -1
        if not dtype.is_int64() or dtype.is_uint64():
            type_width = int(type_width/2)

        var wrote = _snprintf_scalar[dtype, "%f"](
            buf.unsafe_ptr(),
            size,
            value,
        )
        var str_slice = StringSlice[False, __lifetime_of(buf)](
            unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=wrote
        )
        var str_len = len(str(str_slice))

        if str_len < max_width:
            var pad_len = (int(max_width) - str_len) - (type_width)
            pad = pad * int(pad_len)
            var pad_slice = StringSlice[False, __lifetime_of(pad)](
                unsafe_from_utf8_ptr=pad.unsafe_uint8_ptr(), len=pad_len
            )
            writer.write_str(pad_slice)

        writer.write_str(str_slice)


@always_inline("nodebug")
fn TensorPrinter[type : DType, print_type : Bool = False, print_shape : Bool = False](ptr : DTypePointer[type], shape : shape, inout writer : Formatter):
    var rank = shape.rank

    writer.write_str(TensorStart)
    if shape.rank <= 1:
        if shape.rank == 1:
            writer.write_str(SquareBracketL)
            complete[type](ptr, shape.num_elements, writer)
            writer.write_str(SquareBracketR)
        if shape.rank == 0:
            writer.write_str(SquareBracketL+SquareBracketR)

    else:
        var column_elem_count  = 1 if rank < 1 else shape.Shapes()[-1]
        var row_elem_count = 1 if rank < 2 else shape.Shapes()[-2]

        var matrix_elem_count = column_elem_count * row_elem_count
        
        for _ in range(2,rank):
            writer.write_str(SquareBracketL)

        var num_matrices = 1

        for i in range(max(rank -2, 0)):
            num_matrices *= shape.Shapes()[i]
        
        var matrix_idx = 0
        while matrix_idx < num_matrices:
            if matrix_idx > 0:
                writer.write_str(",\n\n\t")
            writer.write_str(SquareBracketL)

            var row_idx = 0
            while row_idx < row_elem_count:
                if row_idx > 0 and row_elem_count > CompactMaxElemsToPrint:
                    writer.write_str("\n\t ")

                if row_idx > 0 and row_elem_count <= CompactMaxElemsToPrint:
                    writer.write_str("\n\t ")

                _serialize_elements[type](
                ptr + matrix_idx * matrix_elem_count + row_idx * column_elem_count,
                column_elem_count, writer)
                row_idx += 1

                if row_idx != row_elem_count:
                    writer.write_str(", ")

                if (row_elem_count >= CompactMaxElemsToPrint and row_idx == CompactElemPerSide):
                    writer.write_str("\n\t")
                    writer.write_str(Truncation)
                    row_idx = row_elem_count - CompactElemPerSide
                
            writer.write_str(SquareBracketR)
            matrix_idx+=1
            if (num_matrices >= CompactMaxElemsToPrint and matrix_idx == CompactElemPerSide):
                writer.write_str("\n\n\t")
                writer.write_str(" ...")
                matrix_idx = num_matrices - CompactElemPerSide

        for _ in range(2,rank):
            writer.write_str(SquareBracketR)

    if print_type:
        var buf = (",  dtype: " + type.__repr__())
        var typeslice = StringSlice[False, __lifetime_of(buf)](unsafe_from_utf8_ptr=buf.unsafe_uint8_ptr(), len=len(buf))
        writer.write_str(typeslice)

    if print_shape:
        var buf = (",  shape: "+shape.__repr__())
        var shapeslice = StringSlice[False, __lifetime_of(buf)](unsafe_from_utf8_ptr=buf.unsafe_uint8_ptr(), len=len(buf))
        writer.write_str(shapeslice)

    writer.write_str(TensorEnd)
    return


@value
struct __PrinterOptions:
    var precision : Int
    var threshold : FloatLiteral
    var edgeitems : Int
    var linewidth : Int
    var sci_mode: Optional[Bool]
    
    fn __init__(inout self):
        self.precision = 4
        self.threshold = 1000
        self.edgeitems = 3
        self.linewidth = 80
        self.sci_mode = None

alias PRINT_OPTS = __PrinterOptions()

#TODO: still under construction...
struct TensorFormatter:
    var floating_dtype : Bool
    var int_mode : Bool
    var sci_mode : Bool
    var max_width : Int

    fn __init__[T : DType](inout self, tensor : Tensor[T], sci_mode : Bool = False):
        self.floating_dtype = tensor.type.is_floating_point()
        self.int_mode = tensor.type.is_integral()
        self.sci_mode = sci_mode
        self.max_width = 1
        
        var tensor_view = tensor.list()

        if not self.floating_dtype:
            for value in tensor_view:
                var value_str = str(value[])
                self.max_width = max(self.max_width, len(value_str))