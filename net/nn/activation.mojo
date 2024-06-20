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



@value
struct Sigmoid[type : DType]:

    @always_inline("nodebug")
    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.

        `Formula:`\n
            `Sigmoid(x) =  1 / (1 + exp(-x))`.

        Arguments:
            Input: Tensor for which Sigmoid activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the sigmoid activation function.
        """
        return sigmoid[type](Input)

    @always_inline("nodebug")
    fn backward(inout self, Input : Tensor[type], GradOutput : Tensor[type]) -> Tensor[type]:
        """
        Compute the gradient of the Sigmoid function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the sigmoid must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the sigmoid.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var SigmoidOutput = self.forward(Input)
        return GradOutput * (SigmoidOutput * (1 - SigmoidOutput))


@value
struct GeLU[type : DType]:

    @always_inline("nodebug")
    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.

        `Formula:`\n
            `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return gelu[type](Input)

    @always_inline("nodebug")
    fn backward(inout self, Input : Tensor[type], GradOutput : Tensor[type]) -> Tensor[type]:
        """
        Compute the gradient of the GeLU function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the GeLU must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the GeLU.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var x = Input
        var sqrt2 : Scalar[type] = math.sqrt(2)
        var erf_comp = (x / sqrt2).apply[math.erf]()
        var exp_comp = ((-(x * x) / Scalar[type](2)).apply[math.exp]())        
        return GradOutput * (0.5 * (1 + erf_comp) + x * (pi * 0.5) * exp_comp / (sqrt2 * sqrt2 * sqrt2 * sqrt2))


@value
struct ReLU[type : DType]:

    @always_inline("nodebug")
    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.

        `Formula:`\n
            `ReLU(x) = max(0, x)`.

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return relu[type](Input)

    @always_inline("nodebug")
    fn backward(inout self, Input : Tensor[type], GradOutput : Tensor[type]) -> Tensor[type]:
        """
        Compute the gradient of the ReLU function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the ReLU must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the ReLU.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var grad = Tensor[type](Input.shapes())
        var num_elements = Input.num_elements()

        @parameter
        fn calc(index: Int):
            var grad_val = GradOutput.load(index)
            if Input.load(index) > 0:
                grad.store(index, grad_val)
            else:
                grad.store(index, 0)

        parallelize[calc](num_elements)
        return grad

@value
struct Tanh[type : DType]:

    @always_inline("nodebug")
    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.

        `Formula:`\n
            `Tanh(x) = (exp(2 * x) - 1) / (exp(2 * x) + 1)`.

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return tanh[type](Input)

    @always_inline("nodebug")
    fn backward(inout self, Input : Tensor[type], GradOutput : Tensor[type]) -> Tensor[type]:
        """
        Compute the gradient of the Tanh function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the Tanh must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the Tanh.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var tanh_x = tanh[type](Input)
        return (1 - tanh_x * tanh_x) * GradOutput


@value
struct SiLU[type : DType]:

    @always_inline("nodebug")
    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.

        `Formula:`\n
            `SiLU(x) = x * sigmoid(x)`.

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return silu[type](Input)

    @always_inline("nodebug")
    fn backward(inout self, Input : Tensor[type], GradOutput : Tensor[type]) -> Tensor[type]:
        """
        Compute the gradient of the SiLU function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the SiLU must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the SiLU.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var sigmoid_x = sigmoid[type](Input)
        return GradOutput * (Input + sigmoid_x * (1 + Input * (1 - sigmoid_x)))


@value
struct Fuctional[type : DType]:

    fn GeLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.
        """
        return gelu[type](Input)

    fn ReLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.
        """
        return relu[type](Input)

    fn SiLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.
        """
        return silu[type](Input)

    fn TanH(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.
        """
        return tanh[type](Input)

    fn Sigmoid(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.
        """
        return sigmoid[type](Input)