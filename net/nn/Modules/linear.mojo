from net.tensor import Tensor

struct Linear[dtype : DType]:
    """
    Linear layer.

    Attributes:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        weights (Tensor): Tensor storing the weights of the layer (output_features, input_features).
        biases (Tensor): Tensor storing the biases of the layer (output_features).
    """
    var Weights : Tensor[dtype]
    var biases : Tensor[dtype]
    var Input_dim : Int
    var Output_dim : Int

    fn __init__(inout self, Input_dim : Int, Output_dim : Int):
        self.Input_dim = Input_dim
        self.Output_dim = Output_dim
        self.Weights = Tensor[dtype](self.Output_dim,self.Input_dim)
        self.biases = Tensor[dtype](self.Output_dim)

        self.Weights.rand()
        self.biases.rand()
    
    fn forward(inout self, inout Inputs : Tensor[dtype], use_bias : Bool = True) -> Tensor[dtype]:
        """
        Applies the linear transformation to the input tensor.

        Formula:
            y = x @ w.T + b

        Args:
            Inputs   : (Tensor[dtype]) The input tensor of shape (batch_size, input_features).
            use_bias : (Bool)  Whether to add the bias term or not.

        Returns:
            Tensor[dtype]: The output tensor of shape (batch_size, output_features).
        """
        var weights_transposed = self.Weights.transposed()
        var y = Inputs.matmul(weights_transposed)
        if use_bias:
            y.add(self.biases)
        return y