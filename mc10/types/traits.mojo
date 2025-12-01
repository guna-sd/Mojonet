from pathlib import Path
from builtin.value import Defaultable
from builtin.simd import Floorable, CeilDivable, Ceilable


trait TensorLike(
    Boolable,
    Copyable,
    Movable,
    CeilDivable,
    Ceilable,
    Comparable,
    Defaultable,
    Floatable,
    Floorable,
    Stringable,
    Writable,
    Powable,
    Representable,
    Roundable,
    Sized,
    Serializable,
):
    alias Type: DType

    fn rank(self) -> Int:
        """Returns the rank (number of dimensions) of the tensor."""
        ...

    fn num_elements(self) -> Int:
        """Returns the total number of elements in the tensor."""
        ...

    fn flatten(self) -> Self:
        """Returns a 1-dimensional version of the tensor."""
        ...

    fn dtype(self) -> DType:
        """Returns the data type of the tensor elements."""
        ...

    fn shape(self) -> List[Int]:
        """Returns the shape of the tensor."""
        ...

    fn reshape(self, shape: List[Int]) -> Self:
        """Returns a new tensor with the specified shape."""
        ...

    fn transpose(self) -> Self:
        """Returns a new tensor that is the transpose of the original tensor."""
        ...


# TODO: Not there yet...
trait Symbolic(
    Absable,
    Boolable,
    CeilDivable,
    Ceilable,
    Copyable,
    Movable,
    Comparable,
    Floorable,
    Hashable,
    ImplicitlyBoolable,
    Intable,
    KeyElement,
    Representable,
    Roundable,
    FloatableRaising,
    IntableRaising,
):
    fn is_int(self) -> Bool:
        ...

    fn is_bool(self) -> Bool:
        ...

    fn is_float(self) -> Bool:
        ...

    fn is_symbol(self) -> Bool:
        ...


trait Serializable:
    fn save(self, path: Path):
        """Saves the object to the specified path."""
        ...

    @staticmethod
    fn load(path: Path) -> Self:
        """Loads the object from the specified path."""
        ...
