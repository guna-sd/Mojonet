import net

trait Module:

    fn __init__(inout self):
        """
        Initialize the layer with necessary parameters.
        """
        ...

    fn forward[dtype : DType](inout self, x : net.Tensor[dtype]) -> net.Tensor[dtype]:
        """
        Forward Layer.

        Parameters:
            dtype: DType of the input data (Tensor).

        Arguments:
            x : Tensor (inputs of the layer).

        Returns:
            Tensor output of the layer.
        """
       ...

    fn backward(inout self):
        """
        Backward Layer.
        """
       ...

    fn weights(inout self):
        """Learnable parameters of the model."""
       ...

    fn parameters(inout self):
        ...

    fn train(inout self):
        """
        Sets the model in training mode.

        This method should set internal flags or states 
        to indicate the model is in training mode.
        """
        ...

    fn eval(inout self):
        """
        Sets the model in evaluation mode.

        This method should set internal flags or states 
        to indicate the model is in evaluation mode.
        """
        ...