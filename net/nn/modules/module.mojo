from net import Tensor

trait layer:
    fn __init__(inout self,):
       ...
    fn forward[dtype : DType](inout self, x : Tensor[dtype]) -> Tensor[dtype]:
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
        """Learnable parameters of the layer."""
       ...
