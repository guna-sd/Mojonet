@value
@register_passable
struct StorageImpl[T: DType]:
    alias element_type = Scalar[T]
    var raw_ptr: Arc[UnsafePointer[Scalar[T]]]
    """Pointer to the stored data."""

    var numel: Int
    """The size of the allocated memory block in number of elements."""

    var device: Device

    fn __init__(inout self):
        """
        Default initializer that sets an empty storage.
        Initializes the pointer with a default empty `DataPointer` and sets size to 0.
        """
        self.raw_ptr = Arc(UnsafePointer[Scalar[T]]())
        self.numel = 0
        self.device = Device.CPU

    fn __init__(
        inout self,
        numel: Int,
        device: Device = Device.CPU,
    ):
        """
        Allocates storage with the specified size, device, and data type.
        """
        var _ptr = Allocator.allocate[Self.element_type](numel, device)
        self.raw_ptr = Arc(_ptr)
        self.numel = numel
        self.device = Device.CPU

    fn __bool__(self) -> Bool:
        if self.numel == 0:
            return False
        return True

    fn load[
        width: Int = 1,
    ](ref [_]self: Self, offset: Int) -> SIMD[T, width]:
        if offset < 0 or offset >= self.numel:
            handle_issue("LoadError: Offset out of bounds.")
        return self.raw_ptr[].load[width=width](offset)

    fn unsafe_load(
        ref [_]self: Self, offset: Int
    ) -> ref [__lifetime_of(self)] Self.element_type:
        if offset < 0 or offset >= self.numel:
            handle_issue("LoadError: Offset out of bounds.")
        return (self.raw_ptr[] + offset)[]

    fn store[
        width: Int = 1,
    ](ref [_]self, offset: Int, owned value: SIMD[T, width]):
        if offset < 0 or offset >= self.numel:
            handle_issue("StoreError: Offset out of bounds.")
            self.raw_ptr[].store(offset, value)

    fn alloc(inout self, numel: Int, dtype: DType, device: Device = Device.CPU):
        self.raw_ptr = Arc(Allocator.allocate[Self.element_type](numel, device))
        self.numel = numel

    fn resize(inout self, new_numel: Int):
        Allocator.reallocate(self.raw_ptr[], new_numel, self.numel)
        self.numel = new_numel

    fn unsafe_ptr(ref [_]self) -> UnsafePointer[Self.element_type]:
        return self.raw_ptr[]

    fn free(owned self):
        Allocator.deallocate(self)
    
    fn cast[Type: DType](ref[ImmutableAnyLifetime]self) -> StorageImpl[Type]:
        var new_ptr = Allocator.allocate[Scalar[Type]](self.numel, self.device)
        for i in range(0, self.numel):
            new_ptr.store(i, self.load(i).cast[Type]())
        var storage = StorageImpl(new_ptr, self.numel, self.device)
        return storage^