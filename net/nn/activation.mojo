@value
struct Sigmoid[type: DType]:
    @always_inline("nodebug")
    fn forward(inout self, Input: Tensor[type]) -> Tensor[type]:
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
    fn backward(
        inout self, Input: Tensor[type], GradOutput: Tensor[type]
    ) -> Tensor[type]:
        """
        Compute the gradient of the Sigmoid function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the sigmoid must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the sigmoid.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var SigmoidOutput = self.forward(Input)
        return GradOutput * (SigmoidOutput * (Scalar[type](1) - SigmoidOutput))


@value
struct GeLU[type: DType]:
    @always_inline("nodebug")
    fn forward(inout self, Input: Tensor[type]) -> Tensor[type]:
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
    fn backward(
        inout self, Input: Tensor[type], GradOutput: Tensor[type]
    ) -> Tensor[type]:
        """
        Compute the gradient of the GeLU function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the GeLU must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the GeLU.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var x = Input
        var sqrt2: Scalar[type] = math.sqrt(2)
        var erf_comp = (x / sqrt2).apply[math.erf]()
        var exp_comp = ((-(x * x) / Scalar[type](2)).apply[math.exp]())
        return GradOutput * (
            Scalar[type](0.5) * (Scalar[type](1) + erf_comp)
            + x
            * (Scalar[type](3.1415926535897931) * Scalar[type](0.5))
            * exp_comp
            / (sqrt2 * sqrt2 * sqrt2 * sqrt2)
        )


@value
struct ReLU[type: DType]:
    @always_inline("nodebug")
    fn forward(inout self, Input: Tensor[type]) -> Tensor[type]:
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
    fn backward(
        inout self, Input: Tensor[type], GradOutput: Tensor[type]
    ) -> Tensor[type]:
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
struct Tanh[type: DType]:
    @always_inline("nodebug")
    fn forward(inout self, Input: Tensor[type]) -> Tensor[type]:
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
    fn backward(
        inout self, Input: Tensor[type], GradOutput: Tensor[type]
    ) -> Tensor[type]:
        """
        Compute the gradient of the Tanh function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the Tanh must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the Tanh.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var tanh_x = tanh[type](Input)
        return (Scalar[type](1) - tanh_x * tanh_x) * GradOutput


@value
struct SiLU[type: DType]:
    @always_inline("nodebug")
    fn forward(inout self, Input: Tensor[type]) -> Tensor[type]:
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
    fn backward(
        inout self, Input: Tensor[type], GradOutput: Tensor[type]
    ) -> Tensor[type]:
        """
        Compute the gradient of the SiLU function with respect to the input tensor.

        Args:
            Input: The input tensor for which the gradient of the SiLU must be calculated.
            GradOutput: The gradient of the loss function with respect to the output of the SiLU.

        Returns:
            Tensor[type]: The gradient of the loss function with respect to the input tensor.
        """
        var sigmoid_x = sigmoid[type](Input)
        return GradOutput * (
            Input
            + sigmoid_x
            * (Scalar[type](1) + Input * (Scalar[type](1) - sigmoid_x))
        )
