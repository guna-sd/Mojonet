"""
Checkpoint: Save and load model parameters.
"""

from .tensor import Tensor
from .linear import Linear


fn save(layer: Linear, path: String) raises:
    """Save Linear layer parameters to file."""
    var file = open(path, "w")

    # Write dimensions
    file.write(
        String(layer.in_features) + " " + String(layer.out_features) + "\n"
    )

    # Write weight values
    for i in range(layer.weight.size()):
        file.write(String(layer.weight[i]) + "\n")

    # Write bias values
    for i in range(layer.bias.size()):
        file.write(String(layer.bias[i]) + "\n")

    file.close()


fn load_into(mut layer: Linear, path: String) raises:
    """Load parameters directly into a Linear layer."""
    var file = open(path, "r")
    var content = file.read()
    file.close()

    var lines = content.split("\n")

    # Parse dimensions
    var dims = lines[0].split(" ")
    var in_f = atol(dims[0])
    var out_f = atol(dims[1])

    # Verify dimensions match
    if in_f != layer.in_features or out_f != layer.out_features:
        raise Error("Checkpoint dimensions don't match layer")

    # Load weights
    var weight_size = in_f * out_f
    for i in range(weight_size):
        layer.weight[i] = Float32(atof(lines[1 + i]))

    # Load bias
    for i in range(out_f):
        layer.bias[i] = Float32(atof(lines[1 + weight_size + i]))
