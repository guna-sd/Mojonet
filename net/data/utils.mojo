from net.utils.utilities import Status

struct Batch[T: CollectionElement]:
    """
    The `Batch` struct represents a collection of data points in a dataset.
    """
    var data: List[T]
    """A list to store the data points in the batch."""

    fn __init__(inout self: Self, ):
        self.data = List[T]()

    fn __init__(inout self: Self, arg: List[T]):
        self.data = arg
    
    fn __init__(inout self: Self, arg: T):
        self.data = List[T](arg)
    
    fn __getitem__(self, index: Int) -> T:
        return self.data[index]
    
    fn __len__(self) -> Int:
        return self.data.__len__()

    fn add(inout self: Self, value: T):
        """
        Adds a new data point to the batch.
        """
        self.data.append(value)

    fn clear(inout self: Self):
        """
        Clears all data points from the batch.
        """
        self.data.clear()