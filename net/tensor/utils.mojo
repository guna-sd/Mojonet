from builtin.io import _snprintf_scalar
from collections.string import _calc_format_buffer_size

alias TensorStart = "Tensor("
alias TensorEnd = ")"
alias SquareBracketL = "["
alias SquareBracketR = "]"
alias Truncation = " ...,"
alias CompactMaxElemsToPrint = 19
alias CompactElemPerSide = 4

@value
struct shape(Movable, Representable, Formattable, Stringable, Sized):
    alias maxrank: Int = 26
    var shapes : List[Int]
    """`shapes:` shapes of ndimensional tensors."""
    var strides : List[Int]
    """`strides:` strides between ndimensional tensors."""
    var layout: Layout
    """`layout:` layout of the tensor."""
    var is_contiguous : Bool
    """`is_contiguous:` This field indicates whether the tensor is contiguous or not."""
    var rank : Int
    """`rank:` The number of dimensions in the tensor, also known as its rank."""
    var num_elements : Int
    """`num_elements:` The total number of elements that a tensor can hold based on the shape."""

    @always_inline("nodebug")
    fn __init__(inout self):
        self.shapes = List[Int]()
        self.strides = List[Int]()
        self.layout = Layout.Strided
        self.is_contiguous = True
        self.num_elements = 0
        self.rank = 0

    @always_inline("nodebug")
    fn __init__(inout self, owned shapes : List[Int]):
        if (len(shapes) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = shapes
        self.strides = calculate_strides(self.shapes)
        self.layout = Layout.Strided
        self.is_contiguous = True
        self.num_elements = num_elements(self.shapes)
        self.rank = len(shapes)

    @always_inline("nodebug")
    fn __init__(inout self, owned shapes : VariadicList[Int]):
        if (len(shapes) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = list(shapes)
        self.strides = calculate_strides(self.shapes)
        self.layout = Layout.Strided
        self.is_contiguous = True
        self.num_elements = num_elements(self.shapes)
        self.rank = len(shapes)

    @always_inline("nodebug")
    fn __init__(inout self, owned *shapes : Int):
        if (len(shapes) > self.maxrank):
            print("WARNING: max rank exceeded")
            print("number of elements must be equal or less then rank {26}")
            exit(1)
        self.shapes = List[Int]()
        for i in range(len(shapes)):
            self.shapes[i] = shapes[i]
        self.strides = calculate_strides(self.shapes)
        self.layout = Layout.Strided
        self.is_contiguous = True
        self.num_elements = num_elements(self.shapes)
        self.rank = len(shapes)
    
    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
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
        var output = String()
        var writer = output._unsafe_to_formatter()
        self.format_to(writer)
        return output^

    @no_inline
    fn format_to(self, inout writer: Formatter):
        writer.write("(")
        for i in range(len(self)):
            writer.write(str(self[i]))
            if i != len(self) - 1:
                writer.write(", ")
        writer.write(")")

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
    fn offset(self : Self, indices : List[Int]) -> Int:
      """Calculates the flat index for a list of multi-dimensional indices."""
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
    
    fn flatten(self) -> shape:
        var new = shape(List[Int](self.num_elements), List[Int](1), self.layout, self.is_contiguous, 1, self.num_elements)
        return new
    
    fn reshape(self: Self, new_shapes: List[Int]) -> shape:
        var new_shape = shape(new_shapes)
        if new_shape.num_elements != self.num_elements:
            handle_issue("shapes should be the same size")
        return new_shape
    
    fn transpose(self) -> shape:
        if self.rank <= 1:
            return self
        var transposed_shapes = List[Int]()
        for i in range(self.rank):
            transposed_shapes[i] = self.shapes[self.rank - i - 1]
        var transposed_strides = List[Int]()
        var new_stride = 1
        for i in range(self.rank - 1, -1, -1):
            transposed_strides[i] = new_stride
            new_stride *= self.shapes[i]
        var new_is_contiguous = self.is_contiguous and (self.rank == 2) and (self.shapes[0] == self.num_elements)

        return shape(
            transposed_shapes, 
            transposed_strides,
            self.layout, 
            new_is_contiguous, 
            self.rank, 
            self.num_elements
            )

@always_inline("nodebug")
fn calculate_strides(shapes : List[Int]) -> List[Int]:
    var stride = 1
    var strides = List[Int]()
    for i in range(shapes.__len__() - 1, -1, -1):
        strides[i] = stride
        stride *= shapes[i]
    return strides


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
fn list[size : Int](shapes : IndexList[size], capacity : Int)-> List[Int]:
    var list = List[Int](capacity=capacity)
    for i in range(capacity):
        list.append(shapes[i])
    return list^

@always_inline("nodebug")
fn list[T: CollectionElement](*shapes : T)-> List[T]:
    var list = List[T](capacity=shapes.__len__())
    for i in shapes:
        list.append(i[])
    return list^

@always_inline("nodebug")
fn bytes[type : DType](num_elements : Int) -> Int:
  var bytes = sizeof[type]()
  return (bytes * num_elements)

# ===-------------------------------------------------------------------------------=== #
# Utilities for Serialising the elements of a tensor into a string representation
# ===-------------------------------------------------------------------------------=== #

@always_inline("nodebug")
fn complete[type : DType,](ptr: UnsafePointer[Scalar[type]], len: Int, inout writer : Formatter):
    """
    Concatenates the elements of a tensor into a string, separated by commas, rounded, and formatted based on the specified width."""
    if len == 0:
        return
    _format_scalar[type](writer, rebind[Scalar[type]](ptr.load[width=1]()))
    for i in range(1,len):
        writer.write(", ")
        _format_scalar[type](writer, rebind[Scalar[type]](ptr.load[width=1](i)))


@always_inline("nodebug")
fn _serialize_elements[type : DType,](ptr: UnsafePointer[Scalar[type]], len: Int, inout writer : Formatter):
    """
    Serializes the elements of a tensor into a string representation, including square brackets.
    """
    writer.write(SquareBracketL)
    var maxelemstoprint: Int = CompactMaxElemsToPrint
    if type is DType.int32 or type is DType.uint32:
        maxelemstoprint = 15
    if type is DType.int64 or type is DType.uint64:
        maxelemstoprint = 9

    var maxelemsperside: Int = CompactElemPerSide
    if type is DType.int64 or type is DType.uint64:
        maxelemsperside = 3
    if len == 0:
        writer.write(SquareBracketR)
        return

    if len < maxelemstoprint:
        complete[type](ptr, len, writer)
        writer.write(SquareBracketR)
        return

    complete[type](ptr, maxelemsperside, writer)
    writer.write(", ")
    writer.write(Truncation)
    complete[type](ptr + len - maxelemsperside, maxelemsperside, writer)

    writer.write(SquareBracketR)
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
        var str_slice = StringSlice[lifetime=__lifetime_of(buf)](
            unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=wrote
        )
        writer.write_str(str_slice)
    else:
        var max_width = len(str(Scalar[dtype].MAX_FINITE))
        var pad = str(" ")
        var type_width = int(dtype.sizeof()/2) -1

        if dtype is DType.int64 or dtype is DType.uint64:
            type_width = type_width -1
        if not dtype is DType.int64 or dtype is DType.uint64:
            type_width = int(type_width/2)

        var wrote = _snprintf_scalar[dtype, "%f"](
            buf.unsafe_ptr(),
            size,
            value,
        )
        var str_slice = StringSlice[lifetime=__lifetime_of(buf)](
            unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=wrote
        )
        var str_len = len(str(str_slice))

        if str_len < max_width:
            var pad_len = (int(max_width) - str_len) - (type_width)
            pad = pad * int(pad_len)
            var pad_slice = StringSlice[lifetime=__lifetime_of(pad)](
                unsafe_from_utf8_ptr=pad.unsafe_ptr(), len=pad_len
            )
            writer.write_str(pad_slice)

        writer.write_str(str_slice)


@always_inline("nodebug")
fn TensorPrinter[type : DType, // , print_type : Bool = False, print_shape : Bool = False](ptr : UnsafePointer[Scalar[type]], shape : shape, inout writer : Formatter):
    var rank = shape.rank

    writer.write(TensorStart)
    if shape.rank <= 1:
        if shape.rank == 1:
            writer.write(SquareBracketL)
            complete[type](ptr, shape.num_elements, writer)
            writer.write(SquareBracketR)
        if shape.rank == 0:
            writer.write(SquareBracketL+SquareBracketR)

    else:
        var column_elem_count  = 1 if rank < 1 else shape.Shapes()[-1]
        var row_elem_count = 1 if rank < 2 else shape.Shapes()[-2]

        var matrix_elem_count = column_elem_count * row_elem_count
        
        for _ in range(2,rank):
            writer.write(SquareBracketL)

        var num_matrices = 1

        for i in range(max(rank -2, 0)):
            num_matrices *= shape.Shapes()[i]
        
        var matrix_idx = 0
        while matrix_idx < num_matrices:
            if matrix_idx > 0:
                writer.write(",\n\n\t")
            writer.write(SquareBracketL)

            var row_idx = 0
            while row_idx < row_elem_count:
                if row_idx > 0 and row_elem_count > CompactMaxElemsToPrint:
                    writer.write("\n\t ")

                if row_idx > 0 and row_elem_count <= CompactMaxElemsToPrint:
                    writer.write("\n\t ")

                _serialize_elements[type](
                ptr + matrix_idx * matrix_elem_count + row_idx * column_elem_count,
                column_elem_count, writer)
                row_idx += 1

                if row_idx != row_elem_count:
                    writer.write(", ")

                if (row_elem_count >= CompactMaxElemsToPrint and row_idx == CompactElemPerSide):
                    writer.write("\n\t")
                    writer.write(Truncation)
                    row_idx = row_elem_count - CompactElemPerSide
                
            writer.write(SquareBracketR)
            matrix_idx+=1
            if (num_matrices >= CompactMaxElemsToPrint and matrix_idx == CompactElemPerSide):
                writer.write("\n\n\t")
                writer.write(" ...")
                matrix_idx = num_matrices - CompactElemPerSide

        for _ in range(2,rank):
            writer.write(SquareBracketR)

    if print_type:
        var buf = (",  dtype: " + type.__repr__())
        var typeslice = StringSlice[lifetime=__lifetime_of(buf)](unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=len(buf))
        writer.write_str(typeslice)

    if print_shape:
        var buf = (",  shape: "+shape.__repr__())
        var shapeslice = StringSlice[lifetime=__lifetime_of(buf)](unsafe_from_utf8_ptr=buf.unsafe_ptr(), len=len(buf))
        writer.write_str(shapeslice)

    writer.write(TensorEnd)
    return


@value
struct __PrinterOptions:
    var precision : Int
    var threshold : FloatLiteral
    var edgeitems : Int
    var linewidth : Int
    var max_width : Int
    var sci_mode: Bool

    fn __init__(inout self, sci_mode: Bool):
        self.precision = 4
        self.threshold = 1000
        self.edgeitems = 3
        self.linewidth = 80
        self.max_width = 1
        self.sci_mode = sci_mode


#TODO: still under construction...
@value
struct TensorFormatter[Type: DType]:
    var Format: __PrinterOptions
    var tensor: Pointer[type=Tensor[Type], lifetime=ImmutableAnyLifetime]


    fn __init__(inout self, tensor : Pointer[Tensor[Type], ImmutableAnyLifetime], sci_mode : Bool = False):
        self.Format = __PrinterOptions(sci_mode)
        self.tensor = tensor
    
    @no_inline
    fn format_to(self, inout writer: Formatter):
        writer.write(TensorStart)
        if self.tensor[].shape[].rank <=1:
            if self.tensor[].shape[].rank == 1:
                writer.write(SquareBracketL)
                complete(self.tensor[].data.raw_ptr[], self.tensor[].shape[].num_elements, writer)
                writer.write(SquareBracketR)
            if self.tensor[].shape[].rank == 0:
                writer.write(SquareBracketL+SquareBracketR)

    fn __format_scalar(self, inout writer: Formatter):
        ...

    fn __format_int(self, inout writer: Formatter):
        ...
    
    fn __format_float(self, inout writer: Formatter):
        ...
    
    fn __format_double(self, inout writer: Formatter):
        ...
    
    fn __format_1d(self, inout writer: Formatter):
        ...