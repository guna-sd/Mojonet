fn add[type : DType](owned first: Scalar[type], owned second: Scalar[type]) -> Scalar[type]:
    """
    Implementation of addition of integer.
    """

    while second != 0:
        var c = first & second
        first ^= second
        second = c << 1
    return first

fn add[type : DType, nelts : Int](first: SIMD[type,nelts], second : SIMD[type,nelts]) -> SIMD[type,nelts]:
    var result = SIMD[type,nelts]()
    @parameter
    fn addition[nelts : Int](i : Int):
        result[i] = first[i] + second[i]
    vectorize[addition,nelts](nelts)
    return result

fn sub[type : DType, nelts : Int](first: SIMD[type,nelts], second : SIMD[type,nelts]) -> SIMD[type,nelts]:
    var result = SIMD[type,nelts]()
    @parameter
    fn subtract[nelts : Int](i : Int):
        result[i] = first[i] - second[i]
    vectorize[subtract,nelts](nelts)
    return result

fn mul[type : DType, nelts : Int](first: SIMD[type,nelts], second : SIMD[type,nelts]) -> SIMD[type,nelts]:
    var result = SIMD[type,nelts]()
    @parameter
    fn multiply[nelts : Int](i : Int):
        result[i] = first[i] * second[i]
    vectorize[multiply,nelts](nelts)
    return result

fn div[type : DType, nelts : Int](first: SIMD[type,nelts], second : SIMD[type,nelts]) -> SIMD[type,nelts]:
    var result = SIMD[type,nelts]()
    @parameter
    fn divide[nelts : Int](i : Int):
        result[i] = first[i] / second[i]
    vectorize[divide,nelts](nelts)
    return result

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