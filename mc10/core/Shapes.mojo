from collections import OptionalReg
from memory import UnsafePointer
from mc10.utils.index import index, indexable
from mc10.utils.debuggable import abort, asserts
from mc10.__mlir import _mlirtype_is_eq

from utils import StaticTuple


@value
@register_passable("trivial")
struct dim:
    """
    A tensor dimension that may be static (known at compile time) or dynamic (unknown).

    When the underlying `static` field is set (i.e. has a value), the dimension is static.
    Otherwise it is dynamic.
    """

    alias dynamic = index(-1)
    var static: OptionalReg[index]

    fn __init__(out self):
        self.static = None

    @implicit
    fn __init__(out self, value: Int):
        self.static = index(value)
        if value == -1:
            self.static = None

    @implicit
    fn __init__(out self, value: index):
        self.static = value
        if value == -1:
            self.static = None

    @always_inline("nodebug")
    fn is_dynamic(self) -> Bool:
        """Determines if the current dim is dynamic.

        Returns:
            Bool: True if the dim is dynamic, False otherwise.
        """
        return not Bool(self.static)

    @always_inline("nodebug")
    fn as_index(self) -> index:
        """Converts the current dim to an index.

        Returns:
            Index: The index representation of the dim.
        """
        return Self.dynamic if self.is_dynamic() else self.static.value()

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Converts the current dim to an integer.

        Returns:
            Int: The integer representation of the dim.
        """
        return (
            Self.dynamic.__int__() if self.is_dynamic() else self.static.value().__int__()
        )

    @always_inline("nodebug")
    fn __as_int__(self) -> Int:
        """Converts the current dim to an integer.

        Returns:
            Int: The integer representation of the dim.
        """
        return Int(self)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Checks if the current dim is equal to another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if both dims are equal, False otherwise.
        """
        return self.as_index() == rhs.as_index()

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Checks if the current dim is not equal to another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if both dims are not equal, False otherwise.
        """
        return self.as_index() != rhs.as_index()

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Checks if the current dim is less than another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if the current dim is less than the other, False otherwise.
        """
        return self.as_index() < rhs.as_index()

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Checks if the current dim is greater than another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if the current dim is greater than the other, False otherwise.
        """
        return self.as_index() > rhs.as_index()

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value of the current dim.

        Returns:
            UInt: The hash value of the dim.
        """
        return self.as_index().__hash__()

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Converts the current dim to a boolean value.

        Returns:
            Bool: The boolean representation of the dim.
        """
        return Bool(self.static)

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        """Converts the current dim to a boolean value.

        Returns:
            Bool: The boolean representation of the dim.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        """Adds the current dim to another dim.

        If either dim is dynamic, the result is dynamic.

        Args:
            rhs : The other dim to add.

        Returns:
            Self: The result of the addition.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return Self()
        else:
            return Self(self.static.value() + rhs.static.value())

    @always_inline("nodebug")
    fn __add__(self, rhs: index) -> Self:
        """Adds the current dim to an index.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to add.

        Returns:
            Self: The result of the addition.
        """
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() + rhs)

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        """Increments the current dim by another dim.

        If either dim is dynamic, no operation is performed.

        Args:
            rhs : The other dim to add.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return
        else:
            self = Self(self.static.value() + rhs.static.value())

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: index):
        """Increments the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to add.
        """
        if self.is_dynamic():
            return
        self = Self(self.static.value() + rhs)

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Subtracts another dim from the current dim.

        If either dim is dynamic, the result is dynamic.

        Args:
            rhs : The other dim to subtract.

        Returns:
            Self: The result of the subtraction.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return Self()
        else:
            return Self(self.static.value() - rhs.static.value())

    @always_inline("nodebug")
    fn __sub__(self, rhs: index) -> Self:
        """Subtracts an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to subtract.

        Returns:
            Self: The result of the subtraction.
        """
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() - rhs)

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        """Decrements the current dim by another dim.

        If either dim is dynamic, no operation is performed.

        Args:
            rhs : The other dim to subtract.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return
        else:
            self = Self(self.static.value() - rhs.static.value())

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: index):
        """Decrements the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to subtract.
        """
        if self.is_dynamic():
            return
        self = Self(self.static.value() - rhs)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Multiplies another dim from the current dim.

        If either dim is dynamic, the result is dynamic.

        Args:
            rhs : The other dim to multiply.

        Returns:
            Self: The result of the product.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return Self()
        else:
            return Self(self.static.value() * rhs.static.value())

    @always_inline("nodebug")
    fn __mul__(self, rhs: index) -> Self:
        """Multiplies an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to multiply.

        Returns:
            Self: The result of the product.
        """
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() * rhs)

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        """Inplace mutiplies the current dim by another dim.

        If either dim is dynamic, no operation is performed.

        Args:
            rhs : The other dim to multiply.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return
        else:
            self = Self(self.static.value() * rhs.static.value())

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: index):
        """Inplace mutiplies the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to multiply.
        """
        if self.is_dynamic():
            return
        self = Self(self.static.value() * rhs)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        """Divides another dim from the current dim.

        If either dim is dynamic, the result is dynamic.

        Args:
            rhs : The other dim to divide.

        Returns:
            Self: The result of the division.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return Self()
        else:
            return Self(self.static.value() / rhs.static.value())

    @always_inline("nodebug")
    fn __truediv__(self, rhs: index) -> Self:
        """Divides an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to divide.

        Returns:
            Self: The result of the division.
        """
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() / rhs)

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Self):
        """Inplace divides the current dim by another dim.

        If either dim is dynamic, no operation is performed.

        Args:
            rhs : The other dim to divide.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return
        else:
            self = Self(self.static.value() / rhs.static.value())

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: index):
        """Inplace divides the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to divide.
        """
        if self.is_dynamic():
            return
        self = Self(self.static.value() / rhs)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Performs floor division of another dim from the current dim.

        If either dim is dynamic, the result is dynamic.

        Args:
            rhs : The other dim to floor divide.

        Returns:
            Self: The result of the floor division.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return Self()
        else:
            return Self(self.static.value() // rhs.static.value())

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: index) -> Self:
        """Performs floor division of an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to floor divide.

        Returns:
            Self: The result of the floor division.
        """
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() // rhs)

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Self):
        """Inplace floor divides the current dim by another dim.

        If either dim is dynamic, no operation is performed.

        Args:
            rhs : The other dim to floor divide.
        """
        if self.is_dynamic() or rhs.is_dynamic():
            return
        else:
            self = Self(self.static.value() // rhs.static.value())

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: index):
        """Inplace floor divides the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to floor divide.
        """
        if self.is_dynamic():
            return
        self = Self(self.static.value() // rhs)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """
        Converts the Dim to a String. If the value is unknown, then the string "?" is returned.

        Returns:
        The string representation of the type.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this DimList to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        if self.is_dynamic():
            writer.write("?")
        else:
            writer.write(self.static.value())


@value
@register_passable("trivial")
struct shape:
    """
    Represents a tensor shape as a collection of dimensions.
    """

    var dims: VariadicList[dim]

    fn __init__(out self):
        self.dims = VariadicList[dim]()

    fn __init__(out self, *dims: dim):
        self.dims = dims

    fn __init__(out self, dims: VariadicList[dim]):
        self.dims = dims

    fn __init__[
        size: Int,
        /,
    ](out self, dims: StaticTuple[dim, size]):
        self = Self.unknown[size]()

        @parameter
        for i in range(size):
            self[i] = dims[i]

    # TODO: either this is not the right way to do this or something is wrong with the implementation
    # fn __init__[*Ts: CollectionElement](out self, dims: Tuple[*Ts]):
    #     asserts(
    #         _mlirtype_is_eq[
    #             VariadicList(Ts)._mlir_type,
    #             __mlir_type[`!kgen.variadic<`, dim, `>`],
    #         ](),
    #         "tuple does not contain dim",
    #     )

    #     alias size = VariadicList(dims.element_types).__len__()

    #     self = Self.unknown[size]()

    #     @parameter
    #     for i in range(size):
    #         print(rebind[dim](dims[i]))
    #         self[i] = rebind[dim](dims[i])

    fn __len__(self) -> Int:
        return len(self.dims)

    @always_inline("nodebug")
    fn __getitem__(self, index: index) -> dim:
        return self.dims[indexable["shape"](index.__int__(), self)]

    @always_inline("nodebug")
    fn __setitem__(mut self, index: index, dimension: dim):
        tmp = UnsafePointer.address_of(self.dims).bitcast[List[dim]]()[]
        tmp[indexable["shape"](index, self)] = dimension

    @always_inline("nodebug")
    fn __setitem__(self, index: index, dimension: Int):
        tmp = UnsafePointer.address_of(self.dims).bitcast[List[dim]]()[]
        tmp[indexable["shape"](index, self)] = dim(dimension)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline("nodebug")
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    @always_inline("nodebug")
    fn __repr__(self: Self) -> String:
        var buf = String("")

        if self.rank() == 0:
            return buf^

        if len(self) == 1:
            buf.write("1x")
            buf.write(self[0])
            return buf^

        for i in range(len(self)):
            if i > 0:
                buf.write("x")
            buf.write(self[i])
        return buf^

    @always_inline("nodebug")
    fn __str__(self: Self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("(")
        for i in range(len(self)):
            writer.write(self[i])
            if i != len(self) - 1:
                writer.write(", ")
        writer.write(")")

    @parameter
    @always_inline("nodebug")
    fn rank(self) -> index:
        """Returns the rank (number of dimensions)."""
        return __mlir_op.`pop.variadic.size`(self.dims.value)

    @staticmethod
    fn unknown[rank: index]() -> Self:
        return Self(
            __mlir_op.`pop.variadic.splat`[
                _type = VariadicList[dim]._mlir_type, numElements = rank.value
            ](dim())
        )

    @always_inline("nodebug")
    fn all_known(self) -> Bool:
        for i in range(len(self)):
            if self[i].is_dynamic():
                return False
        return True

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        """Returns the number of elements based on the given shape."""
        var nelms = 1
        for i in range(len(self)):
            nelms *= self[i]
        return nelms

    @always_inline("nodebug")
    fn reshape(self: Self, *shapes: dim) -> shape:
        var new = shape(shapes)
        if new.num_elements() != self.num_elements():
            abort("shapes should be the same size")
        return new

    @always_inline("nodebug")
    fn list(self) -> List[dim]:
        var list = List[dim](capacity=self.__len__())
        for i in self.dims:
            list.append(i)
        return list^
