from net.utils import shape

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

fn _rank0(type : DType, shape : shape) -> String:
    return TensorStart+SquareBracketL+SquareBracketR+Comma+Strdtype+str(type)+Comma+Strshape+shape.__str__()+TensorEnd

fn complete(ptr : DTypePointer, len : Int) -> String:
    var buf = String("")
    if len == 0:
        return buf
    buf += ptr.load()
    for i in range(1, len):
        buf += ", "
        buf += str(ptr.load(i))
    return buf

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