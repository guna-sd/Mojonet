@value
struct Fuctional[type: DType]:
    fn GeLU(inout self, Input: Tensor[type]) -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.
        """
        return gelu[type](Input)

    fn ReLU(inout self, Input: Tensor[type]) -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.
        """
        return relu[type](Input)

    fn SiLU(inout self, Input: Tensor[type]) -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.
        """
        return silu[type](Input)

    fn TanH(inout self, Input: Tensor[type]) -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.
        """
        return tanh[type](Input)

    fn Sigmoid(inout self, Input: Tensor[type]) -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.
        """
        return sigmoid[type](Input)
