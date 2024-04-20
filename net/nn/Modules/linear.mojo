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
        self.biases.zeros()
    
    fn forward(inout self, Inputs : Tensor[dtype], bias : Bool = True) -> Tensor[dtype]:
        """
        Applies the linear transformation to the input tensor.

        Formula:
            y = x @ w.T + b

        Args:
            Inputs   : (Tensor[dtype]) The input tensor of shape (batch_size, input_features).
            bias : (Bool)  Whether to add the bias term or not.

        Returns:
            Tensor[dtype]: The output tensor of shape (batch_size, output_features).
        """
        if Inputs.shape[1] != self.Input_dim:
            print("Inputs must have the same shape as self.Input_dim (batch_size, input_features).")

        var y = Inputs @ (self.Weights.transposed())
        if bias:
            return y.add(self.biases)
        return y