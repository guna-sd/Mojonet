from mc10.types.DataPointer import DataPointer
from mc10.utils.debuggable import abort


@value
@register_passable("trivial")
struct MemoryBlock[
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = 1,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    var ptr: DataPointer[
        address_space=address_space, alignment=alignment, mut=mut, origin=origin,
    ]
    var size: Int
    var used: Bool

    fn __init__(out self, size: Int):
        self.size = size
        self.ptr = None
        self.used = False

    fn allocate(mut self):
        if not self.used and not self.ptr:
            self.ptr = __type_of(self.ptr).alloc(self.size)
            self.used = True
        else:
            abort("Memory block is already in use.")


@value
@register_passable("trivial")
struct MemoryPool:
    ...
