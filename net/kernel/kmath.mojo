from net.tensor import Tensor

fn abs[type : DType](value : SIMD[type,1]) -> SIMD[type,1]:
    """
    Find the absolute value of a number.
    """
    return -value if value < 0.0 else value

fn max[type : DType](value : Tensor[type]) -> SIMD[type,1]:
    """
    Find the maximum value in the Tensor.
    """
    if value.num_elements() == 0:
        print("is empty")
        return 0
    if value.num_elements() == 1:
        return value[0]
    var j = value[0]
    for i in range(value.num_elements()):
        if abs[type](value[i]) > abs[type](j):
            j = value[i]
    return j


fn min[type : DType](value : Tensor[type]) -> SIMD[type,1]:
    """
    Find the minimum value in the list.
    """
    if value.num_elements() == 0:
        print("is empty")
        return 0
    if value.num_elements() == 1:
        return value[0]

    var j = value[0]
    for i in range(value.num_elements()):
        if abs(value[i]) < abs(j):
            j = value[i]
    return j

fn add[type : DType](owned first: SIMD[type,1], owned second: SIMD[type,1]) -> SIMD[type,1]:
    """
    Implementation of addition of integer.
    """

    while second != 0:
        var c = first & second
        first ^= second
        second = c << 1
    return first

fn add[type : DType](owned first: Tensor[type], owned second: Tensor[type]) -> Tensor[type]:
    var out : Tensor[type] = Tensor[type](first.shape)
    if first.shape._rank == second.shape._rank:
        for i in range(first.shape.num_elements):
            out[i] = add[type](first[i], second[i])
        return out
    else:
        print("adding two different tensors is not supported")
        return out

fn pow[type : DType, nelts : Int](base: SIMD[type,nelts], exponent: Int) -> SIMD[type,nelts]:
    """
    Computes a^b recursively, where a is the base and b is the exponent.
    """
    if exponent < 0:
        print("Exponent must be a non-negative integer")

    if exponent == 0:
        return 1

    if exponent % 2 == 1:
        return pow(base, exponent - 1) * base

    var b = pow(base, exponent // 2)
    return b * b

fn multiply[type : DType](owned a: SIMD[type,1], owned b: SIMD[type,1]) -> SIMD[type,1]:
    """
    Multiply 'a' and 'b' using bitwise multiplication.
    """
    var res : SIMD[type,1] = 0
    while b > 0:
        if b & 1:
            res += a

        a += a
        b >>= 1

    return res

