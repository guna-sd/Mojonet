import net
from collections.dict import Dict
from gpu.host import Device


trait Modules:
    fn forward[
        dtype: DType
    ](inout self, x: net.Tensor[dtype]) -> net.Tensor[dtype]:
        """
        Forward Layer.
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
        """Returns an iterator over module parameters."""
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

    fn zero_grad(inout self):
        """
        Sets gradients of all model parameters to zero.
        """
        ...

    fn to(inout self, device: Device):
        """
        Moves all model parameters to a specified device.
        """
        ...

    fn state_dict[dtype: DType](inout self) -> Dict[String, net.Tensor[dtype]]:
        """
        Returns a dictionary containing a whole state of the module.
        """
        ...

    fn load_state_dict[
        dtype: DType
    ](inout self, state_dict: Dict[String, net.Tensor[dtype]]):
        """
        Copies parameters and buffers from state_dict into this module
        and its descendants.
        """
        ...

    fn train_mode(inout self, mode: Bool):
        """
        Sets the training mode of the module.
        """
        ...

    fn requires_grad_(inout self, requires_grad: Bool):
        """
        Sets whether or not gradients should be computed for model parameters.
        """
        ...

    fn add_module(inout self, name: String, module: Module):
        """
        Adds a child module to the current module.
        """
        ...

    fn to_string(inout self) -> String:
        """
        Returns a string representation of the module.
        """
        ...

    fn save(inout self, filepath: String):
        """
        Saves the module's state to a file.
        """
        ...

    fn load(inout self, filepath: String):
        """
        Loads the module's state from a file.
        """
        ...


@value
struct ModuleType[type: DType]:
    var name: String
    var function: Function


@value
struct Module[type: DType](CollectionElement):
    var modules: Dict[String, ModuleType[type]]
    var params: Dict[String, Parameter[type]]
    var grads: Dict[String, Tensor[type]]
    var training: Bool
