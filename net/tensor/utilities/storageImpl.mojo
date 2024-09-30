@value
@register_passable
struct StorageImpl:
    var ptr: Arc[DataPointer]
    """Pointer to the stored data."""

    var numel: Int
    """The size of the allocated memory block in number of elements."""

    var dtype: DType
    """The datatype of the elements stored in the memory (e.g., float32, int64)."""

    fn __init__(inout self):
        """
        Default initializer that sets an empty storage.
        Initializes the pointer with a default empty `DataPointer` and sets size to 0.
        """
        self.ptr = Arc(DataPointer())
        self.numel = 0
        self.dtype = DType.invalid

    fn __init__(
        inout self,
        numel: Int,
        dtype: DType,
        device: Device = Device.CPU,
    ):
        """
        Allocates storage with the specified size, device, and data type.
        """
        var _ptr = Allocator.allocate(numel, device)
        self.ptr = Arc(_ptr)
        self.numel = numel
        self.dtype = dtype

    fn __bool__(self) -> Bool:
        if self.numel == 0 and self.dtype == DType.invalid:
            return False
        return True

    fn load[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](ref [_]self: Self, offset: Int) -> SIMD[type, width]:
        if offset < 0 or offset >= self.numel:
            handle_issue("LoadError: Offset out of bounds.")
        return (self.ptr[].load[type=type, width=width, alignment=alignment](offset))

    fn unsafe_load[type: DType](ref [_]self: Self, offset: Int) -> ref [__lifetime_of(self)] Scalar[type]:
        if offset < 0 or offset >= self.numel:
            handle_issue("LoadError: Offset out of bounds.")
        return (self.ptr[].address.bitcast[type]().__getitem__(offset))

    fn store[
        type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = 1,
    ](ref [_]self, offset: Int, owned value: SIMD[type, width]):
        if offset < 0 or offset >= self.numel:
            handle_issue("StoreError: Offset out of bounds.")
        (self.ptr[].store[type=type, width=width, alignment=alignment](offset, value))

    fn alloc(inout self, numel: Int, dtype: DType, device: Device = Device.CPU):
        self.ptr = Arc(Allocator.allocate(numel, device))
        self.numel = numel
        self.dtype = dtype

    fn resize(inout self, new_numel: Int):
        Allocator.reallocate(self.ptr[], new_numel, self.numel)
        self.numel = new_numel
    
    fn unsafe_ptr(ref [_]self) -> UnsafePointer[UInt8]:
        return self.ptr[].address
    
    fn data_ptr(ref [_]self) -> DataPointer:
        return self.ptr[]
    
    fn free(inout self):
        self = Self()