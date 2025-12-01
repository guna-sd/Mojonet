from collections import OptionalReg
from mc10.utils.debuggable import abort, asserts


# This is a underdeveloped code might look like a good way to store a static but not there yet...

@register_passable("trivial")
struct dim(Intable, Writable):
    """
    A tensor dimension that may be static (known at compile time) or dynamic (unknown).

    When the underlying `static` field is set (i.e. has a value), the dimension is static.
    Otherwise it is dynamic.
    """

    alias dynamic = Int(-1111)._mlir_value
    var static: OptionalReg[Int]

    fn __init__(out self):
        self.static = None

    @implicit
    fn __init__(out self, value: Int):
        self.static = value

    @always_inline("nodebug")
    fn is_dynamic(self) -> Bool:
        """Determines if the current dim is dynamic.

        Returns:
            Bool: True if the dim is dynamic, False otherwise.
        """
        return self.static is None

    @always_inline("nodebug")
    fn as_index(self) -> __mlir_type.index:
        """Converts the current dim to an index.

        Returns:
            Index: The index representation of the dim.
        """
        return (
            Self.dynamic if self.is_dynamic() else self.static.value()._mlir_value
        )

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Converts the current dim to an integer.

        Returns:
            Int: The integer representation of the dim.
        """
        return Int(
            mlir_value=Self.dynamic
        ) if self.is_dynamic() else self.static.value()

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
        return Int(self) == Int(rhs)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Checks if the current dim is not equal to another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if both dims are not equal, False otherwise.
        """
        return Int(self) == Int(rhs)

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Checks if the current dim is less than another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if the current dim is less than the other, False otherwise.
        """
        return Int(self) < Int(rhs)

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Checks if the current dim is greater than another dim.

        Args:
            rhs : The other dim to compare with.

        Returns:
            Bool: True if the current dim is greater than the other, False otherwise.
        """
        return Int(self) > Int(rhs)

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Converts the current dim to a boolean value.

        Returns:
            Bool: The boolean representation of the dim.
        """
        return self.is_dynamic()

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
    fn __add__(self, rhs: Int) -> Self:
        """Adds the current dim to an index.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to add.

        Returns:
            Self: The result of the addition.
        """
        if self.is_dynamic():
            return Self()
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
    fn __iadd__(mut self, rhs: Int):
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
    fn __sub__(self, rhs: Int) -> Self:
        """Subtracts an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to subtract.

        Returns:
            Self: The result of the subtraction.
        """
        if self.is_dynamic():
            return Self()
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
    fn __isub__(mut self, rhs: Int):
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
    fn __mul__(self, rhs: Int) -> Self:
        """Multiplies an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to multiply.

        Returns:
            Self: The result of the product.
        """
        if self.is_dynamic():
            return Self()
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
    fn __imul__(mut self, rhs: Int):
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
            return Self(Int(self.static.value() / rhs.static.value()))

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Int) -> Self:
        """Divides an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to divide.

        Returns:
            Self: The result of the division.
        """
        if self.is_dynamic():
            return Self()
        return Self(Int(self.static.value() / rhs))

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
            self = Self(Int(self.static.value() / rhs.static.value()))

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Int):
        """Inplace divides the current dim by an index.

        If the dim is dynamic, no operation is performed.

        Args:
            rhs : The index to divide.
        """
        if self.is_dynamic():
            return
        self = Self(Int(self.static.value() / rhs))

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
    fn __floordiv__(self, rhs: Int) -> Self:
        """Performs floor division of an index from the current dim.

        If the dim is dynamic, the result is dynamic.

        Args:
            rhs : The index to floor divide.

        Returns:
            Self: The result of the floor division.
        """
        if self.is_dynamic():
            return Self()
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
    fn __ifloordiv__(mut self, rhs: Int):
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


# Deprecated... Needs a fix...
# TODO: Move for a much cleaner way...

# @register_passable("trivial")
# struct shape[rank: OptionalReg[Int]](Sized, Writable):
#     """
#     Represents a tensor shape as a collection of dimensions.
#     """

#     comptime is_comptime: Bool = Self.rank != None
#     var _runtime_value: InlineArray[DType, Self.rank]
#     var dims: VariadicList[dim]

#     fn __init__(out self):
#         self.dims = VariadicList[dim]()

#     fn __init__(out self, *dims: dim):
#         self.dims = dims

#     fn __init__(out self, dims: VariadicList[dim]):
#         self.dims = dims

#     fn __len__(self) -> Int:
#         return len(self.dims)

#     @always_inline("nodebug")
#     fn __getitem__(self, index: Int) -> dim:
#         return self.dims[indexable["shape"](index, self)]

#     @always_inline("nodebug")
#     fn __eq__(self, other: Self) -> Bool:
#         if len(self) != len(other):
#             return False
#         for i in range(len(self)):
#             if self[i] != other[i]:
#                 return False
#         return True

#     @always_inline("nodebug")
#     fn __ne__(self, other: Self) -> Bool:
#         return not self.__eq__(other)

#     @always_inline("nodebug")
#     fn __contains__(self, value: Int) -> Bool:
#         for i in range(len(self)):
#             if self[i] == value:
#                 return True
#         return False

#     @always_inline("nodebug")
#     fn __repr__(self: Self) -> String:
#         var buf = String("")

#         if self.rank() == 0:
#             return buf^

#         if len(self) == 1:
#             buf.write("1x")
#             buf.write(self[0])
#             return buf^

#         for i in range(len(self)):
#             if i > 0:
#                 buf.write("x")
#             buf.write(self[i])
#         return buf^

#     @always_inline("nodebug")
#     fn __str__(self: Self) -> String:
#         return String.write(self)

#     @no_inline
#     fn write_to[W: Writer](self, mut writer: W):
#         writer.write("(")
#         for i in range(len(self)):
#             writer.write(self[i])
#             if i != len(self) - 1:
#                 writer.write(", ")
#         writer.write(")")

#     @parameter
#     @always_inline("nodebug")
#     fn rank(self) -> Int:
#         """Returns the rank (number of dimensions)."""
#         return Int(mlir_value=__mlir_op.`pop.variadic.size`(self.dims.value))

#     @staticmethod
#     fn unknown[rank: Int]() -> Self:
#         return Self(
#             __mlir_op.`pop.variadic.splat`[
#                 _type = VariadicList[dim]._mlir_type,
#                 numElements = rank._mlir_value,
#             ](dim())
#         )

#     @always_inline("nodebug")
#     fn all_known(self) -> Bool:
#         for i in range(len(self)):
#             if self[i].is_dynamic():
#                 return False
#         return True

#     @always_inline("nodebug")
#     fn num_elements(self) -> Int:
#         """Returns the number of elements based on the given shape."""
#         var nelms = 1
#         for i in range(len(self)):
#             nelms *= Int(self[i])
#         return nelms

#     @always_inline("nodebug")
#     fn list(self) -> List[dim]:
#         var list = List[dim](capacity=self.__len__())
#         for i in self.dims:
#             list.append(i)
#         return list^
