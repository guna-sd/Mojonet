from pathlib import Path
from builtin.value import Defaultable
from builtin.simd import Floorable, CeilDivable, Ceilable


trait TensorLike(
    Boolable,
    CollectionElement,
    CeilDivable,
    Ceilable,
    Comparable,
    Defaultable,
    ExplicitlyCopyable,
    Floatable,
    Floorable,
    Stringable,
    Writable,
    Powable,
    Representable,
    Roundable,
    Sized,
):
    alias Type: DType

    fn rank(self) -> Int:
        """Returns the rank (number of dimensions) of the tensor."""
        ...

    fn num_elements(self) -> Int:
        """Returns the total number of elements in the tensor."""
        ...

    fn tofile(self, path: Path):
        """Writes the tensor's contents to a file at the specified path."""
        ...

    fn flatten(self) -> Self:
        """Returns a 1-dimensional version of the tensor."""
        ...

    fn dtype(self) -> DType:
        """Returns the data type of the tensor elements."""
        ...


trait Symbolic(
    Absable,
    Boolable,
    CeilDivable,
    Ceilable,
    CollectionElement,
    Comparable,
    ComparableCollectionElement,
    EqualityComparableCollectionElement,
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


trait Serializable:
    fn save(self, path: Path):
        """Saves the object to the specified path."""
        ...

    @staticmethod
    fn load(path: Path) -> Self:
        """Loads the object from the specified path."""
        ...

