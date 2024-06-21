@always_inline("nodebug")
fn relu[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return max[type, nelts](value, 0)


@always_inline("nodebug")
fn sigmoid[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 1.0 / (1.0 + math.exp[type, nelts](-value))


@always_inline("nodebug")
fn softplus[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.log[type, nelts](1.0 + math.exp[type, nelts](value))


@always_inline("nodebug")
fn swish[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value * sigmoid[type, nelts](value)


@always_inline("nodebug")
fn tanh[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (2 / (1 + math.exp[type, nelts]((-2 * value)))) - 1


@always_inline("nodebug")
fn gelu[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (
        0.5
        * value
        * (
            1.0
            + math.tanh[type, nelts](
                math.sqrt[type, nelts](2.0 / pi)
                * (value + 0.044715 * pow(value, 3))
            )
        )
    )


@always_inline("nodebug")
fn squareplus[
    type: DType, nelts: Int
](value: SIMD[type, nelts], beta: SIMD[type, 1]) -> SIMD[type, nelts]:
    return (value + math.sqrt[type, nelts](value**2 + beta)) / 2


@always_inline("nodebug")
fn tanh[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """Function `tanh`: apply hyperbolic tangent activation to given Tensor.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor with the hyperbolic tangent of the input tensor elements applied.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn op[nelts: Int](n: Int):
        Output.store[nelts](n, math.tanh[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = math.tanh(Input[n])

    return Output


@always_inline("nodebug")
fn sigmoid[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor where each element is transformed by the sigmoid function.
    """

    alias nelts = simdwidthof[type]() * 2
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shapes())

    @parameter
    fn op[nelts: Int](n: Int):
        Output.store[nelts](n, sigmoid[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = sigmoid(Input[n])

    return Output


@always_inline("nodebug")
fn relu[type: DType](Input: Tensor[type]) -> Tensor[type]:
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
    fn op[nelts: Int](n: Int):
        Output.store[nelts](n, relu[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = relu(Input[n])

    return Output


@always_inline("nodebug")
fn gelu[type: DType](Input: Tensor[type]) -> Tensor[type]:
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
    fn op[nelts: Int](n: Int):
        Output.store[nelts](n, gelu[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = gelu(Input[n])

    return Output


@always_inline("nodebug")
fn silu[type: DType](Input: Tensor[type]) -> Tensor[type]:
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
    fn op[nelts: Int](n: Int):
        Output.store[nelts](n, swish[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = swish(Input[n])

    return Output
