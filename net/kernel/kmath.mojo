import math

alias SELU_ALPHA = 1.6732632423543772848170429916717
alias SELU_SCALE = 1.0507009873554804934193349852946

@always_inline("nodebug")
fn erfc[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.erfc(arg)


@always_inline("nodebug")
fn erf[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.erf(arg)


@always_inline("nodebug")
fn j0[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.j0(arg)


@always_inline("nodebug")
fn sin[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.sin(arg)


@always_inline("nodebug")
fn sinh[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.sinh(arg)


@always_inline("nodebug")
fn cosh[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.cosh(arg)


@always_inline("nodebug")
fn cos[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.cos(arg)


@always_inline("nodebug")
fn atan[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.atan(arg)


@always_inline("nodebug")
fn tan[type: DType, nelts: Int](arg: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.tan(arg)


@always_inline("nodebug")
fn relu[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return max[type, nelts](value, 0)


@always_inline("nodebug")
fn sigmoid[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 1.0 / (1.0 + math.exp[type, nelts](-value))


@always_inline("nodebug")
fn hard_sigmoid[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return ((value + 1) / 2).clamp(0, 1)


@always_inline("nodebug")
fn softplus[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.log[type, nelts](1.0 + math.exp[type, nelts](value))


@always_inline("nodebug")
fn mish[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value * (tanh[type, nelts](softplus[type, nelts](value)))


@always_inline("nodebug")
fn swish[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value * sigmoid[type, nelts](value)


@always_inline("nodebug")
fn hard_swish[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    var offset = 3.0
    var scale = 1.0 / 6.0
    return value * ((value + offset).clamp(0, offset)) * scale


@always_inline("nodebug")
fn tanh[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (2.0 / (1.0 + math.exp[type, nelts]((-2.0 * value)))) - 1.0


@always_inline("nodebug")
fn hard_tanh[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value.clamp(-1, 1)


@always_inline("nodebug")
fn arctan[
    type: DType, nelts: Int
](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return atan[type, nelts](value)


@always_inline("nodebug")
fn gelu[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (
        0.5
        * value
        * (
            1.0
            + tanh[type, nelts](sqrthfpi * (value + 0.044715 * pow(value, 3)))
        )
    )


@always_inline("nodebug")
fn softmax[
    type: DType, nelts: Int
](logits: SIMD[type, nelts]) -> SIMD[type, nelts]:
    var max_val = max[type, nelts](logits, 0)
    var exp = math.exp[type, nelts](logits - max_val)
    return exp / exp.reduce_add()


@always_inline("nodebug")
fn elu[
    type: DType, nelts: Int
](value: SIMD[type, nelts], alpha: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return max[type, nelts](value, (alpha * (math.exp[type, nelts](value) - 1)))


@always_inline("nodebug")
fn leaky_relu[
    type: DType, nelts: Int
](value: SIMD[type, nelts], alpha: Scalar[type]) -> SIMD[type, nelts]:
    return max[type, nelts](value, (value * alpha))


@always_inline("nodebug")
fn selu[type: DType, nelts: Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return max[type, nelts](
        value, (SELU_SCALE * SELU_ALPHA * (math.exp[type, nelts](value) - 1))
    )


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
        Output.store[nelts](n, tanh[type, nelts](Input.load[nelts](n)))

    vectorize[op, nelts](num_elements - (num_elements % nelts))

    for n in range(num_elements - (num_elements % nelts), num_elements):
        Output[n] = tanh(Input[n])

    return Output^


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

    return Output^


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

    return Output^


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

    return Output^


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

    return Output^
