@value
@register_passable("trivial")
struct Vector[T: AnyTrivialRegType, Vector_size: Int]:
    """
    A dynamic array-like container for storing elements of type T with an initial capacity defined by Vector_size.
    """

    var Storage: UnsafePointer[T]
    """`Storage:`Pointer to the array's storage."""
    var Size: UInt64
    """`Size:`The current number of elements in the vector."""
    var Capacity: UInt64
    """`Capacity:`The current allocated capacity of the vector."""

    fn __init__(inout self):
        self.Storage = UnsafePointer[T]().alloc(Vector_size)
        self.Capacity = Vector_size
        self.Size = 0

    fn __init__(inout self, data: UnsafePointer[T], total_capacity: Int):
        self.Capacity = UInt64(total_capacity)
        self.Storage = data
        self.Size = 0

    fn __init__(inout self, capacity: Int):
        self.Capacity = UInt64(capacity)
        self.Storage = UnsafePointer[T]().alloc(capacity)
        self.Size = 0

    fn __init__(inout self, owned *elements: T):
        self.Storage = UnsafePointer[T]().alloc(len(elements))
        self.Capacity = len(elements)
        self.Size = len(elements)
        for i in range(len(elements)):
            self.Storage[i] = elements[i]

    fn __iter__(
        self: Reference[Self, _, _],
    ) -> _VectortIter[T, Vector_size, self.is_mutable, self.lifetime]:
        return _VectortIter(0, self, True)

    fn __reversed__(
        self: Reference[Self, _, _]
    ) -> _VectortIter[T, Vector_size, self.is_mutable, self.lifetime]:
        return _VectortIter(len(self[]), self, False)

    fn size(self) -> Int:
        return int(self.Size)

    fn capacity(self) -> Int:
        return int(self.Capacity)

    fn clear(inout self):
        """Clears the elements in the list."""
        for i in range(self.Size):
            destroy_pointee(self.Storage + i)
        self.Size = 0

    fn set_size(inout self, n: UInt64):
        debug_assert(
            n <= self.Capacity, "size must be less than or equal to capacity"
        )
        self.Size = n

    fn resize(inout self, min_size: Int, element_size: Int):
        """
        Resize the vector to accommodate at least min_size elements.
        """
        var new_capacity = max(min_size, self.Capacity * 2)
        var new_memory = UnsafePointer[T]().alloc(
            int(new_capacity * element_size)
        )
        memcpy(new_memory, self.Storage, int(self.Size * element_size))
        UnsafePointer[T].free(self.Storage)
        self.Storage = new_memory
        self.Capacity = new_capacity

    fn append(inout self, owned element: T):
        """
        Append an element to the end of the vector.
        """
        if self.size() == self.capacity():
            self.resize(self.size() + 1, sizeof[T]())
        self.Storage[self.size()] = element
        self.Size += 1

    fn pop(inout self, index: Int = -1) -> T:
        """
        Remove and return the element at the given position in the Vector.
        If no index is specified, removes and returns the last element in the Vector.
        """
        debug_assert(-len(self) <= index < len(self), "pop index out of range")
        var normalized_idx = index
        if index < 0:
            normalized_idx += len(self)
        var ret_val = (self.Storage + normalized_idx)[]
        for j in range(normalized_idx + 1, self.Size):
            (self.Storage + j)[] = (self.Storage + j - 1)[]
        self.Size -= 1
        if self.Size * 4 < self.Capacity:
            if self.Capacity > 1:
                self.resize(int(max(self.Capacity // 2, 1)), sizeof[T]())
        return ret_val

    fn index(self, index: Int) -> T:
        return self.Storage[index]

    fn __getitem__(
        self: Reference[Self, _, _], index: Int
    ) -> Reference[T, self.is_mutable, self.lifetime]:
        debug_assert(
            -int(self[].Size) <= index < int(self[].Size),
            "index must be within bounds",
        )
        return (self[].Storage + index)[]

    fn __setitem__(inout self, index: Int, owned value: T):
        self.Storage[index] = value

    fn __len__(self) -> Int:
        return self.size()


@value
struct _VectortIter[
    T: AnyTrivialRegType,
    Size: Int,
    mutable: Bool,
    lifetime: AnyLifetime[mutable].type,
]:
    """An iterator for the Vector struct."""

    var index: Int
    var src: Reference[Vector[T, Size], mutable, lifetime]
    var forward: Bool

    fn __next__(inout self) -> Reference[T, mutable, lifetime]:
        if self.forward:
            self.index += 1
            return self.src[][self.index - 1]
        else:
            self.index -= 1
            return self.src[][self.index]

    fn __len__(self) -> Int:
        if self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index
