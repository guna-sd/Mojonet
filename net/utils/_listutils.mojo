
@value
struct NDArray[type : DType]:
    """
    NDArray is a fixed-size array of N dimensions containing elements of a single data type.

    Attributes:
    shape (tuple of Int): Dimensions of the array, determining the size in each dimension.
    dtype (dtype): The data type of the array's elements.
    size (Int): The total number of elements in the array.
    ndim (Int): The number of dimensions (or axes) of the array.
    strides (tuple of Int): The step sizes to move along each dimension to traverse the array.

    Returns:
    An NDArray object.
    """

    var shape : Tuple[Int]
    var dtype : DType
    var size : Int
    var ndim : Int
    var strides : Tuple[Int]
    var data : DTypePointer[type]

    fn __init__(inout self):
        self.data = DTypePointer[type]().alloc(0)
        self.shape = 0
        self.size = 0
        self.ndim = 0
        self.strides = 0
        self.dtype = type

    # fn __init__(inout self, shape : Tuple[Int]):
    #     """
    #     Initialize an NDArray object.

    #     Parameters:
    #     shape (tuple of Int): The shape of the array.
    #     dtype (dtype, optional): The data type of the array's elements. If not specified, the default data type is used.
    #     """

    #     self.shape = shape
    #     self.ndim = len(shape)
    #     self.data = DTypePointer[type]().alloc(shape)

    #fn __init__(inout self, shape : Tuple[Int], d

    fn __str__(inout self) -> String:
        return "NDArray({self.array}, shape={self.shape}, dtype={self.dtype})"


fn main():
    var lit : List[Int]