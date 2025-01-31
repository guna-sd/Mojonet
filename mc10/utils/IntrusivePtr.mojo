from memory import UnsafePointer


trait RefCounted(AnyType, Movable):
    """
    The `RefCounted` trait is designed for types that require reference counting to manage their lifetime.
    It is intended for use with types that are movable and have a reference count to ensure they are properly
    managed across ownership transfers. The trait defines methods for incrementing and decrementing the reference
    count, and allows for fine-grained control over the moving of instances.

    #### Example:
    ```mojo
    from mc10.utils.IntrusivePtr import RefCounted

    @value
    struct MyObject(RefCounted):
        var refcount: UInt64
        var data: Int

        fn __init__(out self, data: Int):
            self.data = data
            self.refcount = 1

        fn add_ref(mut self):
            self.refcount += 1

        fn drop_ref(mut self) -> Bool:
            self.refcount -= 1
            return self.refcount == 0

        fn count_ref(self) -> UInt64:
            return self.refcount

        fn Unsafe_set_ref(mut self, owned new: UInt64):
            self.refcount = new
    ```

    Methods:

        - `add_ref(mut self)`: Increments the reference count.
        - `drop_ref(mut self) -> Bool`: Decrements the reference count and returns `True` if the reference count reaches zero.
        - `count_ref(self) -> UInt64`: Returns the current reference count.
        - `Unsafe_set_ref(mut self, owned refCount: UInt64)`: Directly sets the reference count.
    """

    fn add_ref(mut self):
        """Increment the refcount."""
        ...

    fn drop_ref(mut self) -> Bool:
        """Decrement the refcount and return true if the result hits zero."""
        ...

    fn count_ref(self) -> UInt64:
        ...

    fn Unsafe_set_ref(mut self, owned refCount: UInt64):
        ...


@register_passable
struct IntrusivePointer[T: RefCounted]:
    """Intrusive reference-counted pointer.

    `IntrusivePointer` is a smart pointer that maintains a reference count for a type `T` that implements the
    `RefCounted` trait. This pointer is designed to manage ownership and memory of the referenced object, ensuring
    that it is automatically cleaned up when no longer needed. The reference count is managed by atomic operations,
    ensuring thread safety in multi-threaded environments.

    When an `IntrusivePointer` is copied, it shares ownership of the referenced object and increments the reference
    count. When the pointer is deleted, it decrements the reference count, and if the reference count reaches zero,
    the object is destroyed and its memory is freed.

    Example:
        ```mojo
        from mc10.utils.IntrusivePtr import IntrusivePointer, RefCounted

        @value
        struct MyObject(RefCounted):
            var refcount: UInt64
            var data: Int

            fn __init__(out self, data: Int):
                self.data = data
                self.refcount = 1

            fn add_ref(mut self):
                self.refcount += 1

            fn drop_ref(mut self) -> Bool:
                self.refcount -= 1
                return self.refcount == 0

            fn count_ref(self) -> UInt64:
                return self.refcount

            fn Unsafe_set_ref(mut self, owned new: UInt64):
                self.refcount = new

        var obj = MyObject(10)
        var ptr = IntrusivePointer(obj^)

        var ptr2 = ptr

        ptr2[].data = 20

        print(ptr.count())  # Output: 2 (both ptr and ptr2 are referencing the same object)
        ```

    Parameters:
        T: The type of the object being referenced.
    """

    var _inner: UnsafePointer[T]

    @implicit
    fn __init__(out self, owned obj: T):
        """Create a new `IntrusivePointer` for the given object.

        This constructor allocates memory for the pointer, initializes the reference count to 1, and stores the
        object in the allocated memory.

        Args:
            obj: The object to manage with the smart pointer.
        """
        self._inner = UnsafePointer[T]().alloc(1)
        obj.Unsafe_set_ref(1)
        self._inner.init_pointee_move(obj^)

    fn copy(self) -> Self:
        """Create a copy of the smart pointer.

        This copies the pointer, incrementing the reference count for the managed object.

        Returns:
            A new `IntrusivePointer` pointing to the same object.
        """
        return self

    fn __copyinit__(out self, existing: Self):
        """Initialize a new pointer by copying from an existing one.

        This method is called when copying an existing `IntrusivePointer`. It increments the reference count of the
        managed object to reflect the new ownership.

        Args:
            existing: The existing `IntrusivePointer` to copy.
        """
        self._inner = existing._inner
        self[].add_ref()

    @no_inline
    fn __del__(owned self):
        """Delete the smart pointer and clean up the managed object.

        This method is called when the `IntrusivePointer` is destroyed. It decrements the reference count for the
        managed object. If the reference count reaches zero, the object is destroyed, and the memory is freed.
        """
        if self._inner:
            if self[].drop_ref():
                self._inner.destroy_pointee()
                self._inner.free()

    fn __getitem__(ref self) -> ref [self] T:
        """Get a mutable reference to the managed object.

        This method returns a mutable reference to the object managed by the `IntrusivePointer`, allowing
        modifications to the underlying object.

        Returns:
            A mutable reference to the managed object.
        """
        return self._inner[]

    fn count(self) -> UInt64:
        """Get the current reference count.

        This method returns the number of active references to the managed object.

        Returns:
            The current reference count.
        """
        return self[].count_ref()

    fn __is__(self, rhs: Self) -> Bool:
        """Check if two `IntrusivePointer` instances point to the same object.

        This method compares two pointers to see if they refer to the same object in memory.

        Args:
            rhs: The other `IntrusivePointer` to compare.

        Returns:
            `True` if both pointers point to the same object, `False` otherwise.
        """
        return self._inner == rhs._inner

    fn __isnot__(self, rhs: Self) -> Bool:
        """Check if two `IntrusivePointer` instances point to different objects.

        This method compares two pointers to see if they refer to different objects in memory.

        Args:
            rhs: The other `IntrusivePointer` to compare.

        Returns:
            `True` if both pointers point to different objects, `False` otherwise.
        """
        return self._inner != rhs._inner
    
    fn __str__(self) -> String:
        return String.write(self)
    
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self._inner)