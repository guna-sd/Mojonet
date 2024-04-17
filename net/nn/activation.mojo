from algorithm import parallelize, vectorize
from sys.info import num_physical_cores
from net.tensor import Tensor
from net.kernel.constant import pi
import math


@always_inline
fn _relu[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.max[type,nelts](value, 0)


@always_inline
fn _sigmoid[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 1.0 / (1.0 + math.exp[type,nelts](-value))


@always_inline
fn _softplus[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return math.log[type, nelts](1.0 + math.exp[type, nelts](value))


@always_inline
fn _swish[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return value * _sigmoid[type, nelts](value)


@always_inline
fn _tanh[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return (2 / (1 + math.exp[type, nelts]((-2 * value)))) - 1


@always_inline
fn _gelu[type : DType, nelts : Int](value: SIMD[type, nelts]) -> SIMD[type, nelts]:
    return 0.5 * value * (1.0 + math.tanh[type, nelts](math.sqrt[type,nelts](2.0 / pi) * (value + 0.044715 * math.pow[type,nelts](value, 3))))


@always_inline
fn _squareplus[type : DType, nelts : Int](value: SIMD[type, nelts], beta : SIMD[type,1]) -> SIMD[type, nelts]:
    return (value + math.sqrt[type, nelts](value**2 + beta)) / 2


fn tanh_vectorized[type : DType](Input : Tensor[type]) -> Tensor[type]:
    """
    Applies tanh to the input -> Uses Vectorization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].

    Returns:
        Input transformed by tanh.
    """
    alias nelts = simdwidthof[type]()
    var num_elements: Int = Input.num_elements()
    var Output: Tensor[type] = Tensor[type](Input.shape)

    @parameter
    fn _row[nelts : Int](row : Int):
        for i in range(num_elements // nelts):
            var offset : Int = row * num_elements + i * nelts
            var value = Input.load[nelts](offset)
            value = math.tanh[type, nelts](value)
            Output.store[nelts](offset, value)
    
    vectorize[_row, nelts](num_elements)
    return Output

    
fn tanh_parallelized[type : DType](Input : Tensor[type], num_cores : Int = 1) -> Tensor[type]:
    """
    Applies tanh to the input -> Uses Parallelization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].
        num_cores : Number of cores to use for Parallelization.

    Returns:
        Input transformed by tanh.
    """
    alias nelts = simdwidthof[type]()
    var num_elements: Int = Input.num_elements()
    var Output: Tensor[type] = Tensor[type](Input.shape)
    var cores = num_cores if num_cores <= num_physical_cores() else 1

    @parameter
    fn _row(size: Int):
        Output[size] = math.tanh(Input[size])

    parallelize[_row](num_elements, cores)
    return Output


fn tanh[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor
    """

    alias nelts = simdwidthof[DType.float32]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn tanh_[nelts: Int](n: Int):
            Output.store[nelts](n,math.tanh(Input[n]))

        vectorize[tanh_,nelts](num_elements)

    parallelize[calc_row](num_elements, 4)
    return Output


fn tanh2[type : DType](Input : Tensor[type], cores : Int = 4, alg : String = "vectorize") -> Tensor[type]:
    """
    Tanh activation function.

    Parameters:
        type: Data type of the tensor.

    Arguments:
        Input: Tensor for which tanh activation function is to be applied.
        cores: Number of cores to use in parallelized implementation (default is 4).
        alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

    Returns:
        Tensor after applying tanh activation function.
    """
    if alg == "vectorize" or alg == "parallelize":
        if Input.num_elements() <= 10000 or num_physical_cores() <= 2 or alg == "vectorize":
            return tanh_vectorized[type](Input)
        return tanh_parallelized[type](Input, cores)
    else:
        print("Invalid algorithm using default algorithm")
        return tanh_vectorized[type](Input)


fn sigmoid_vectorized[type : DType](Input : Tensor[type]) -> Tensor[type]:
    """
    Applies sigmoid 1/ (1 + e^-x) to the Input x. -> Uses Vectorization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].

    Returns:
        Input transformed by sigmoid.
    """
    alias nelts = simdwidthof[type]()
    var num_elements: Int = Input.num_elements()
    var Output: Tensor[type] = Tensor[type](Input.shape)

    @parameter
    fn _row[nelts : Int](row : Int):
        for i in range(num_elements // nelts):
            var offset : Int = row * num_elements + i * nelts
            var value = Input.load[nelts](offset)
            value = _sigmoid(value)
            Output.store[nelts](offset, value)
    
    vectorize[_row, nelts](num_elements)
    return Output


fn sigmoid_parallelized[type : DType = DType.float32](Input : Tensor[type], num_cores : Int = 1) -> Tensor[type]:
    """
    Applies sigmoid 1/ (1 + e^-x) to the Input x. -> Uses Parallelization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].
        num_cores : Number of cores to use for Parallelization.

    Returns:
        Input transformed by sigmoid.
    """
    var num_elements: Int = Input.num_elements()
    var Output: Tensor[type] = Tensor[type](Input.shape)
    var cores = num_cores if num_cores <= num_physical_cores() else 1

    @parameter
    fn _row(size: Int):
        Output[size] = _sigmoid(Input[size])

    parallelize[_row](num_elements, cores)
    return Output


fn sigmoid[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor
    """

    alias nelts = simdwidthof[DType.float32]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn sigmoid_[nelts: Int](n: Int):
            Output.store[nelts](n,_sigmoid[type](Input[n]))

        vectorize[sigmoid_,nelts](num_elements)

    parallelize[calc_row](num_elements, 4)
    return Output


fn sigmoid2[type : DType](Input : Tensor[type], cores : Int = 4, alg : String = "vectorize") -> Tensor[type]:
    """
    Sigmoid activation function 1/ (1 + e^-x).

    Parameters:
        type: Data type of the tensor.

    Arguments:
        Input: Tensor for which Sigmoid activation function is to be applied.
        cores: Number of cores to use in parallelized implementation (default is 4).
        alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

    Returns:
        Tensor after applying sigmoid activation function.
    """
    if alg == "vectorize" or alg == "parallelize":
        if Input.num_elements() <= 10000 or num_physical_cores() <= 2 or alg == "vectorize":
            return sigmoid_vectorized[type](Input)
        return sigmoid_parallelized[type](Input, cores)
    else:
        print("Invalid algorithm using default algorithm")
        return sigmoid_vectorized[type](Input)


fn relu_vectorized[type : DType](Input : Tensor[type]) -> Tensor[type]:
    """
    Applies ReLU to the input -> Uses Vectorization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].

    Returns:
        Input transformed by ReLU.
    """
    alias nelts = simdwidthof[type]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn _row[nelts : Int](row : Int):
        for i in range(num_elements // nelts):
            var offset = row * num_elements + i * nelts
            var value = Input.load[nelts](offset)
            value = _relu[type,nelts](value)
            Output.store[nelts](offset, value)

    vectorize[_row, nelts](num_elements)
    return Output


fn relu_parallelized[type : DType](Input : Tensor[type], num_cores : Int = 1) -> Tensor[type]:
    """
    Applies ReLU to the input -> Uses Parallelization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].
        num_cores : Number of cores to use for Parallelization.

    Returns:
        Input transformed by ReLU.
    """
    var num_elements: Int = Input.num_elements()
    var Output = Tensor[type](Input.shape)
    var cores = num_cores if num_cores <= num_physical_cores() else 1

    @parameter
    fn _row(size: Int):
        Output[size] = _relu[type, 1](Input[size])

    parallelize[_row](num_elements, cores)
    return Output


fn relu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor
    """

    alias nelts = simdwidthof[DType.float32]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn relu_[nelts: Int](n: Int):
            Output.store[nelts](n,_relu[type](Input[n]))

        vectorize[relu_,nelts](num_elements)

    parallelize[calc_row](num_elements, 4)
    return Output


fn relu2[type : DType](Input : Tensor[type], cores : Int = 4, alg : String = "vectorize") -> Tensor[type]:
    """
    ReLU activation function.

    Parameters:
        type: Data type of the tensor.

    Arguments:
        Input: Tensor for which ReLU activation function is to be applied.
        cores: Number of cores to use in parallelized implementation (default is 4).
        alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").
        
    Returns:
        Tensor after applying relu activation function.
    """
    if alg == "vectorize" or alg == "parallelize":
        if Input.num_elements() <= 10000 or num_physical_cores() <= 2 or alg == "vectorize":
            return relu_vectorized[type](Input)
        return relu_parallelized[type](Input, cores)
    else:
        print("Invalid algorithm using default algorithm")
        return relu_vectorized[type](Input)


fn gelu_vectorized[type : DType](Input : Tensor[type]) -> Tensor[type]:
    """
    Applies GELU to the input -> Uses Vectorization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].

    Returns:
        Input transformed by GELU.
    """
    alias nelts = simdwidthof[type]()
    var num_elements = Input.num_elements()
    var Output : Tensor[type] = Tensor[type](Input.shape)

    @parameter
    fn _row[nelts : Int](row : Int):
        for i in range(num_elements // nelts):
            var offset = row * num_elements + i * nelts
            var value = Input.load[nelts](offset)
            value = _gelu[type,nelts](value)
            Output.store[nelts](offset, value)

    vectorize[_row, nelts](num_elements)
    return Output


fn gelu_parallelized[type : DType](Input : Tensor[type], num_cores : Int = 1) -> Tensor[type]:
    """
    Applies GELU to the input -> Uses Parallelization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].
        num_cores : Number of cores to use for Parallelization.

    Returns:
        Input transformed by GELU.
    """
    var num_elements: Int = Input.num_elements()
    var Output = Tensor[type](Input.shape)
    var cores = num_cores if num_cores <= num_physical_cores() else 1

    @parameter
    fn _row(size: Int):
        Output[size] = _gelu(Input[size])

    parallelize[_row](num_elements, cores)
    return Output


fn gelu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor
    """

    alias nelts = simdwidthof[DType.float32]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn gelu_[nelts: Int](n: Int):
            Output.store[nelts](n,_gelu[type](Input[n]))

        vectorize[gelu_,nelts](num_elements)

    parallelize[calc_row](num_elements, 4)
    return Output


fn gelu2[type : DType](Input : Tensor[type], cores : Int = 4, alg : String = "vectorize") -> Tensor[type]:
    """
    GELU activation function.

    Parameters:
        type: Data type of the tensor.

    Arguments:
        Input: Tensor for which GELU activation function is to be applied.
        cores: Number of cores to use in parallelized implementation (default is 4).
        alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

    Returns:
        Tensor after applying GELU activation function.
    """

    if alg == "vectorize" or alg == "parallelize":
        if Input.num_elements() <= 10000 or num_physical_cores() <= 2 or alg == "vectorize":
            return gelu_vectorized[type](Input)
        return gelu_parallelized[type](Input, cores)
    else:
        print("Invalid algorithm using default algorithm")
        return gelu_vectorized[type](Input)


fn silu_parallelized[type : DType](Input : Tensor[type], num_cores : Int = 1) -> Tensor[type]:
    """
    Applies SiLU to the input -> Uses Parallelization algorithm.

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].
        num_cores : Number of cores to use for Parallelization.

    Returns:
        Input transformed by SiLU.
    """
    var num_elements: Int = Input.num_elements()
    var Output = Tensor[type](Input.shape)
    var cores = num_cores if num_cores <= num_physical_cores() else 1

    @parameter
    fn _row(size: Int):
        Output[size] = Input[size] * _sigmoid(Input[size])

    parallelize[_row](num_elements, cores)
    return Output


fn silu_vectorized[type : DType](Input : Tensor[type]) -> Tensor[type]:
    """
    Applies SiLU to the input -> Uses Vectorization algorithm.
    

    Parameters:
        type : DType of the Input data.

    Args:
        Input: Tensor[type].

    Returns:
        Input transformed by SiLU.
    """
    alias nelts : Int = simdwidthof[type]()
    var Output : Tensor[type] = Tensor[type](Input.shape)
    var num_elements : Int = Input.num_elements()

    @parameter
    fn _row[nelts : Int](row : Int):
        for i in range(num_elements // nelts):
            var offset = row * num_elements + i * nelts
            var value = Input.load[nelts](offset)
            value = value * _sigmoid[type,nelts](value)
            Output.store[nelts](offset, value)
    vectorize[_row, nelts](num_elements)
    return Output


fn silu[type : DType](Input: Tensor[type]) -> Tensor[type]:
    """ Function `sigmoid`: apply sigmoid activation to given Tensor.

    Args:
        Input: Input Tensor.
    Returns:
        Tensor
    """

    alias nelts = simdwidthof[DType.float32]()
    var num_elements = Input.num_elements()
    var Output = Tensor[type](Input.shape)

    @parameter
    fn calc_row(m: Int):

        @parameter
        fn silu_[nelts: Int](n: Int):
            Output.store[nelts](n,(Input[n] * _sigmoid[type,nelts](Input[n])))

        vectorize[silu_,nelts](num_elements)

    parallelize[calc_row](num_elements, 4)
    return Output


fn silu2[type : DType](Input : Tensor[type], cores : Int = 4, alg : String = "vectorize") -> Tensor[type]:
    """
    SiLU (Swish) activation function.

    Parameters:
        type: Data type of the tensor.

    Arguments:
        Input: Tensor for which SiLU activation function is to be applied.
        cores: Number of cores to use in parallelized implementation (default is 4).
        alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

    Returns:
        Tensor after applying SiLU activation function.
    """
    if alg == "vectorize" or alg == "parallelize":
        if Input.num_elements() <= 10000 or num_physical_cores() <= 2 or alg == "vectorize":
            return silu_vectorized[type](Input)
        return silu_parallelized[type](Input, cores)
    else:
        print("Invalid algorithm using default algorithm")
        return silu_vectorized[type](Input)


@value
struct Sigmoid[type : DType]:

    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.

        Formula:
            Sigmoid(x) =  1 / (1 + exp(-x)).

        Arguments:
            Input: Tensor for which Sigmoid activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the sigmoid activation function.
        """
        return sigmoid[type](Input)

    fn forward2(inout self, Input : Tensor[type], core : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.

        Formula:
            Sigmoid(x) =  1 / (1 + exp(-x)).

        Arguments:
            Input: Tensor for which Sigmoid activation function is to be applied.
            cores: Number of cores to use in parallelized implementation (default is 4).
            alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

        Returns:
            Tensor[type]: The input tensor after applying the sigmoid activation function.
        """
        return sigmoid[type](Input)


@value
struct GeLU[type : DType]:

    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.

        Formula:
            GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return gelu[type](Input)

    fn forward2(inout self, Input : Tensor[type], core : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.

        Formula:
            GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.
            cores: Number of cores to use in parallelized implementation (default is 4).
            alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return gelu2[type](Input,core,alg)

@value
struct ReLU[type : DType]:

    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.

        Formula:
            ReLU(x) = max(0, x).

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return relu[type](Input)

    fn forward2(inout self, Input : Tensor[type], core : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.

        Formula:
            ReLU(x) = max(0, x).

        Arguments:
            Input: Tensor for which ReLU activation function is to be applied.
            cores: Number of cores to use in parallelized implementation (default is 4).
            alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

        Returns:
            Tensor[type]: The input tensor after applying the ReLU activation function.
        """
        return relu2[type](Input,core,alg)


@value
struct Tanh[type : DType]:

    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.

        Formula:
            Tanh(x) = (exp(2 * x) - 1) / (exp(2 * x) + 1).

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return tanh[type](Input)

    fn forward2(inout self, Input : Tensor[type], core : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.

        Formula:
            Tanh(x) = (exp(2 * x) - 1) / (exp(2 * x) + 1).

        Arguments:
            Input: Tensor for which tanh activation function is to be applied.
            cores: Number of cores to use in parallelized implementation (default is 4).
            alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

        Returns:
            Tensor[type]: The input tensor after applying the tanh activation function.
        """
        return tanh2[type](Input,core,alg)


@value
struct SiLU[type : DType]:

    fn forward(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.

        Formula:
            SiLU(x) = x * sigmoid(x).

        Arguments:
            Input: Tensor for which GELU activation function is to be applied.

        Returns:
            Tensor[type]: The input tensor after applying the GELU activation function.
        """
        return silu[type](Input)

    fn forward2(inout self, Input : Tensor[type], core : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.

        Formula:
            SiLU(x) = x * sigmoid(x).

        Arguments:
            Input: Tensor for which SiLU activation function is to be applied.
            cores: Number of cores to use in parallelized implementation (default is 4).
            alg: Algorithm type ("vectorize" or "parallelize") to choose from (default is "vectorize").

        Returns:
            Tensor[type]: The input tensor after applying the SiLU activation function.
        """
        return silu2[type](Input,core,alg)


@value
struct Fuctional[type : DType]:

    fn GeLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.
        """
        return gelu[type](Input)

    fn GeLU2(inout self, Input : Tensor[type], cores : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor.
        """
        return gelu2[type](Input,cores,alg)

    fn ReLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.
        """
        return relu[type](Input)

    fn ReLU2(inout self, Input : Tensor[type], cores : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the ReLU (Rectified Linear Unit) activation function to the input tensor.
        """
        return relu2[type](Input,cores,alg)

    fn SiLU(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.
        """
        return silu[type](Input)

    fn SiLU2(inout self, Input : Tensor[type], cores : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the SiLU (Sigmoid-Weighted Linear Unit) activation function to the input tensor.
        """
        return silu2[type](Input,cores,alg)

    fn TanH(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.
        """
        return tanh[type](Input)

    fn TanH2(inout self, Input : Tensor[type], cores : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the Tanh (Hyperbolic Tangent) activation function to the input tensor.
        """
        return tanh2[type](Input,cores,alg)

    fn Sigmoid(inout self, Input : Tensor[type]) -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.
        """
        return sigmoid[type](Input)

    fn Sigmoid2(inout self, Input : Tensor[type], cores : Int = 4, alg : String = 'vectorize') -> Tensor[type]:
        """
        Apply the sigmoid activation function to the input tensor.
        """
        return sigmoid2[type](Input,cores,alg)


    