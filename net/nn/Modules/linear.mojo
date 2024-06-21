from net.nn.parameter import parameter


struct Linear[dtype: DType]:
    """
    Applies a linear transformation to the incoming data `y = x @ w.T + b`.

    Args:

        input_features (int): Number of input features.
        output_features (int): Number of output features.
        weights (Tensor): Tensor storing the weights of the layer (output_features, input_features).
        biases (Tensor): Tensor storing the biases of the layer (output_features).
    """

    var Weights: Tensor[dtype]
    var bias: Optional[Tensor[dtype]]
    var Input_dim: Int
    var Output_dim: Int

    fn __init__(inout self, Input_dim: Int, Output_dim: Int, bias: Bool = True):
        self.Input_dim = Input_dim
        self.Output_dim = Output_dim
        self.Weights = parameter[dtype](Output_dim, Input_dim)
        if bias:
            self.bias = parameter[dtype](self.Output_dim, 1)
        else:
            self.bias = None

    @always_inline("nodebug")
    fn forward(inout self, Inputs: Tensor[dtype]) -> Tensor[dtype]:
        """
        Applies the linear transformation to the input tensor.

        `Formula:`\n
            `y = x @ w.T + b`

        Args:
            Inputs   : (Tensor[dtype]) The input tensor of shape (batch_size, input_features).

        Returns:
            Tensor[dtype]: The output tensor of shape (batch_size, output_features).
        """
        if Inputs.tensor.shape[1] != self.Input_dim:
            print(
                "Inputs must have the same shape as Input_dim (batch_size,"
                " input_features)."
            )

        var y = Inputs @ (self.Weights.transposed())

        if self.bias:
            return y.add(self.bias.take())

        return y
