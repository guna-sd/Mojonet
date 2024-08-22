from net.tensor import Tensor
from net.kernel import Constants, Operation
from math import (
    erfc,
    erf,
    j0,
    j1,
    y0,
    y1,
    sinh,
    cosh,
    cos,
    sin,
    tan,
    atan,
    atan2,
    atanh,
    exp,
    log,
    logb,
)

alias SELU_ALPHA = 1.6732632423543772848170429916717
alias SELU_SCALE = 1.0507009873554804934193349852946


@always_inline("nodebug")
fn relu[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `relu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `relu` of the input.
    """
    return max(arg, 0)


@always_inline("nodebug")
fn sigmoid[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `sigmoid` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `sigmoid` of the input.
    """
    return 1.0 / (1.0 + exp(-arg))


@always_inline("nodebug")
fn hard_sigmoid[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `hard_sigmoid` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `hard_sigmoid` of the input.
    """
    return ((arg + 1) / 2).clamp(0, 1)


@always_inline("nodebug")
fn softplus[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `softplus` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `softplus` of the input.
    """
    return log(1.0 + exp(arg))


@always_inline("nodebug")
fn mish[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `mish` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `mish` of the input.
    """
    return arg * (tanh[type, simd_width](softplus[type, simd_width](arg)))


@always_inline("nodebug")
fn swish[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `swish` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `swish` of the input.
    """
    return arg * sigmoid[type, simd_width](arg)


@always_inline("nodebug")
fn hard_swish[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `hard_swish` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `hard_swish` of the input.
    """
    var offset : Scalar[type] = 3.0
    var scale : Scalar[type]  = 1.0 / 6.0
    return (arg * ((arg + offset).clamp(0, offset)) * scale)


@always_inline("nodebug")
fn tanh[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `tanh` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `tanh` of the input.
    """
    return (2.0 / (1.0 + exp((-2.0 * arg)))) - 1.0


@always_inline("nodebug")
fn hard_tanh[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `hard_tanh` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `hard_tanh` of the input.
    """
    return arg.clamp(-1, 1)


@always_inline("nodebug")
fn arctan[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `atanh` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `atanh` of the input.
    """
    return atan(arg)


@always_inline("nodebug")
fn gelu[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `gelu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `gelu` of the input.
    """
    return (
        0.5
        * arg
        * (
            1.0
            + tanh[type, simd_width](
                Constants.sqrthfpi * (arg + 0.044715 * pow(arg, 3))
            )
        )
    )


@always_inline("nodebug")
fn softmax[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `softmax` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `softmax` of the input.
    """
    var max_val = max(arg, 0)
    var exp = exp(arg - max_val)
    return exp / exp.reduce_add()


@always_inline("nodebug")
fn elu[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width], alpha: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """
    Computes the `elu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.
        alpha: The alpha factor for the `ELU` formulation.

    Returns:
    The `elu` of the input.
    """
    return max(
        arg, (alpha * (exp(arg) - 1))
    )


@always_inline("nodebug")
fn leaky_relu[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width], alpha: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """
    Computes the `leaky_relu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.
        alpha: The alpha factor for the `leaky_relu` formulation.

    Returns:
    The `leaky_relu` of the input.
    """
    return max(arg, (arg * alpha))


@always_inline("nodebug")
fn selu[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """
    Computes the `selu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.
        simd_width: The width of the input and output SIMD vector.

    Args:
        arg: The input argument.

    Returns:
    The `selu` of the input.
    """
    return max(
        arg, (SELU_SCALE * SELU_ALPHA * (exp(arg) - 1))
    )


@always_inline("nodebug")
fn elu[type: DType](Input: Tensor[type], alpha: Scalar[type]) -> Tensor[type]:
    """
    Computes the `elu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.
        alpha: The alpha factor for the `ELU` formulation.

    Returns:
    The `elu` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = elu
    var result = Operation[type].tensor_operation[func](Input.tensor, alpha)
    return Tensor[type](result, Input.requires_grad)


@always_inline("nodebug")
fn leaky_relu[
    type: DType
](Input: Tensor[type], alpha: Scalar[type]) -> Tensor[type]:
    """
    Computes the `leaky_elu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.
        alpha: The alpha factor for the `ELU` formulation.

    Returns:
    The `leaky_relu` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width], SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = leaky_relu
    var result = Operation[type].tensor_operation[func](Input.tensor, alpha)
    return Tensor[type](result, Input.requires_grad)


@always_inline("nodebug")
fn selu[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `selu` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `selu` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = selu
    return Input.apply[func]()


@always_inline("nodebug")
fn arctan[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `atan` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `atan` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = arctan
    return Input.apply[func]()


@always_inline("nodebug")
fn hard_sigmoid[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `hard_simoid` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `hard_sigmoid` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = hard_sigmoid
    return Input.apply[func]()


@always_inline("nodebug")
fn hard_tanh[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `hard_tanh` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `hard_tanh` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = hard_tanh
    return Input.apply[func]()


@always_inline("nodebug")
fn softmax[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `softmax` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `softmax` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = softmax
    return Input.apply[func]()


@always_inline("nodebug")
fn softplus[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `softplus` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `softplus` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = softplus
    return Input.apply[func]()


@always_inline("nodebug")
fn mish[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `mish` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `mish` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = mish
    return Input.apply[func]()


@always_inline("nodebug")
fn hard_swish[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Computes the `hard_swish` of the inputs.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: The input argument.

    Returns:
    The `hard_swish` of the input.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = hard_swish
    return Input.apply[func]()


@always_inline("nodebug")
fn tanh[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """Function `tanh`: apply hyperbolic tangent activation to given Tensor.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor with the hyperbolic tangent of the input tensor elements applied.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = tanh
    return Input.apply[func]()


@always_inline("nodebug")
fn sigmoid[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """Function `sigmoid`: apply sigmoid activation to given Tensor.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: Input Tensor.

    Returns:
        A new Tensor where each element is transformed by the sigmoid function.
    """
    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = sigmoid
    return Input.apply[func]()


@always_inline("nodebug")
fn relu[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `relu`: apply ReLU activation to given Tensor.
    ReLU activation is defined as `max(0, x)` for each element x in the Tensor.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with ReLU applied element-wise.
    """

    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = relu
    return Input.apply[func]()


@always_inline("nodebug")
fn gelu[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `gelu`: apply GELU activation to given Tensor.
    GELU activation is defined as `x * Φ(x), where Φ(x)` is the CDF of the standard normal distribution.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with GELU applied element-wise.
    """

    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = gelu
    return Input.apply[func]()


@always_inline("nodebug")
fn silu[type: DType](Input: Tensor[type]) -> Tensor[type]:
    """
    Function `silu`: apply SiLU (Swish) activation to given Tensor.
    SiLU activation is defined as `x * sigmoid(x)` for each element x in the Tensor.

    Parameters:
        type: The dtype of the input and output SIMD vector. Constraints: must be a floating-point type.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor: New Tensor with SiLU applied element-wise.
    """

    alias func: fn[type: DType, simd_width: Int] (
        SIMD[type, simd_width]
    ) -> SIMD[type, simd_width] = swish
    return Input.apply[func]()
