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

@always_inline("nodebug")
fn relu[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return max[type,nelts](value, 0)


@always_inline("nodebug")
fn sigmoid[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 1.0 / (1.0 + math.exp[type,nelts](-value))


@always_inline("nodebug")
fn softplus[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.log[type, nelts](1.0 + math.exp[type, nelts](value))


@always_inline("nodebug")
fn swish[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value * sigmoid[type, nelts](value)


@always_inline("nodebug")
fn tanh[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (2 / (1 + math.exp[type, nelts]((-2 * value)))) - 1


@always_inline("nodebug")
fn gelu[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 0.5 * value * (1.0 + math.tanh[type, nelts](math.sqrt[type,nelts](2.0 / pi) * (value + 0.044715 * pow(value, 3))))


@always_inline("nodebug")
fn squareplus[type : DType, nelts : Int](value: SIMD[type, nelts], beta : SIMD[type,1]) -> SIMD[type, nelts]:
    return (value + math.sqrt[type, nelts](value**2 + beta)) / 2


@always_inline("nodebug")
fn tanh[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `tanh`: apply hyperbolic tangent activation to given Tensor.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor with the hyperbolic tangent of the input tensor elements applied.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn calc_row(`_` : Int):
        @parameter
        fn tanh_op[nelts: Int](n: Int):
            Output.store[nelts](n, math.tanh[type,nelts](Input.load[nelts](n)))

        vectorize[tanh_op, nelts](num_elements - (num_elements % nelts))

        for n in range(num_elements - (num_elements % nelts), num_elements):
            Output[n] = math.tanh(Input[n])

    parallelize[calc_row](Output.shapes()[-2], Output.shapes()[-2])
    return Output


@always_inline("nodebug")
fn sigmoid[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor where each element is transformed by the sigmoid function.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn calc_row(`_` : Int):
        @parameter
        fn sigmoid_op[nelts: Int](n: Int):
            Output.store[nelts](n, sigmoid[type,nelts](Input.load[nelts](n)))

        vectorize[sigmoid_op, nelts](num_elements - (num_elements % nelts))

        for n in range(num_elements - (num_elements % nelts), num_elements):
            Output[n] = sigmoid(Input[n])

    parallelize[calc_row](Output.shapes()[-2], Output.shapes()[-2])
    return Output


@always_inline("nodebug")
fn relu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `relu`: apply ReLU activation to given Tensor.
    ReLU activation is defined as `max(0, x)` for each element x in the Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with ReLU applied element-wise.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn calc_row(`_` : Int):
        @parameter
        fn relu_op[nelts: Int](n: Int):
            Output.store[nelts](n, relu[type,nelts](Input.load[nelts](n)))

        vectorize[relu_op, nelts](num_elements - (num_elements % nelts))

        for n in range(num_elements - (num_elements % nelts), num_elements):
            Output[n] = relu[type, 1](Input[n])

    parallelize[calc_row](Output.shapes()[-2], Output.shapes()[-2])
    return Output


@always_inline("nodebug")
fn gelu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `gelu`: apply GELU activation to given Tensor.
    GELU activation is defined as `x * Φ(x), where Φ(x)` is the CDF of the standard normal distribution.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with GELU applied element-wise.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn calc_row(`_` : Int):
        @parameter
        fn gelu_op[nelts: Int](n: Int):
            Output.store[nelts](n, gelu[type, nelts](Input.load[nelts](n)))
        vectorize[gelu_op, nelts](num_elements - (num_elements % nelts))

        for n in range(num_elements - (num_elements % nelts), num_elements):
            Output[n] = gelu(Input[n])

    parallelize[calc_row](Output.shapes()[-2], Output.shapes()[-2])
    return Output



@always_inline("nodebug")
fn silu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `silu`: apply SiLU (Swish) activation to given Tensor.
    SiLU activation is defined as `x * sigmoid(x)` for each element x in the Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with SiLU applied element-wise.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn calc_row(`_` : Int):
        @parameter
        fn silu_op[nelts: Int](n: Int):
            Output.store[nelts](n,(Input.load[nelts](n) * sigmoid[type,nelts](Input.load[nelts](n))))
        vectorize[silu_op, nelts](num_elements - (num_elements % nelts))

        for n in range(num_elements - (num_elements % nelts), num_elements):
            Output[n] = Input[n] * sigmoid(Input[n])

    parallelize[calc_row](Output.shapes()[-2], Output.shapes()[-2])
    return Output