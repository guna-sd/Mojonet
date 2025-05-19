from mc10.types.traits import Iterable
from memory import UnsafePointer


@value
struct Iterator[
    mut: Bool, //,
    T: Iterable,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    var index: Int
    var forward: Bool
    var src: UnsafePointer[T, mut=mut, origin=origin]

    @always_inline
    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(
        mut self, out p: Pointer[T, origin]
    ):

        if self.forward:
            p = Pointer(to=self.src[self.index])
            self.index += 1
        else:
            self.index -= 1
            p = Pointer(to=self.src[self.index])

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:

        if self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index