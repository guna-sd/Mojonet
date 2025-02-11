from collections import OptionalReg
from memory import UnsafePointer
from mc10.utils.index import index, indexable
from os import abort


@value
@register_passable("trivial")
struct dim:
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
        return not Bool(self.static)

    @always_inline("nodebug")
    fn as_index(self) -> index:
        return Self.dynamic if self.is_dynamic() else self.static.value()

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return (
            Self.dynamic.__int__() if self.is_dynamic() else self.static.value().__int__()
        )

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        return self.as_index() == rhs.as_index()

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        return self.as_index() != rhs.as_index()

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        return self.as_index() < rhs.as_index()

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        return self.as_index() > rhs.as_index()

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        return self.as_index().__hash__()

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return Bool(self.static)

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        return self.__bool__()

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        if self.is_dynamic() and rhs.is_dynamic():
            return Self(None)
        elif not self.is_dynamic() and not rhs.is_dynamic():
            return Self(self.static.value() + rhs.static.value())

        abort("Cannot add a static and dynamic dimension.")
        return Self(None)

    @always_inline("nodebug")
    fn __add__(self, rhs: index) -> Self:
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() + rhs)

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        if self.is_dynamic() and rhs.is_dynamic():
            return Self(None)
        elif not self.is_dynamic() and not rhs.is_dynamic():
            return Self(self.static.value() - rhs.static.value())

        abort("Cannot sub a static and dynamic dimension.")
        return Self(None)

    @always_inline("nodebug")
    fn __sub__(self, rhs: index) -> Self:
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() - rhs)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        if self.is_dynamic() and rhs.is_dynamic():
            return Self(None)
        elif not self.is_dynamic() and not rhs.is_dynamic():
            return Self(self.static.value() * rhs.static.value())

        abort("Cannot mul a static and dynamic dimension.")
        return Self(None)

    @always_inline("nodebug")
    fn __mul__(self, rhs: index) -> Self:
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() * rhs)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        if self.is_dynamic() and rhs.is_dynamic():
            return Self(None)
        elif not self.is_dynamic() and not rhs.is_dynamic():
            return Self(self.static.value() / rhs.static.value())

        abort("Cannot div a static and dynamic dimension.")
        return Self(None)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: index) -> Self:
        if self.is_dynamic():
            return Self(None)
        return Self(self.static.value() / rhs)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self.is_dynamic():
            writer.write("dynamic")
        else:
            writer.write(self.static.value())

    @always_inline("nodebug")
    fn scale(mut self, factor: index):
        if self.is_dynamic():
            abort("Scaling dynamic dimensions is not supported.")
        self = Self(self.static.value() * factor)


@value
@register_passable("trivial")
struct shape:
    var dims: VariadicList[dim]

    fn __init__(out self):
        self.dims = VariadicList[dim]()

    fn __init__(out self, *dims: dim):
        self.dims = dims

    fn __init__(out self, dims: VariadicList[dim]):
        self.dims = dims

    fn __len__(self) -> Int:
        return len(self.dims)

    @always_inline("nodebug")
    fn __getitem__(self, index: index) -> dim:
        return self.dims[indexable["shape"](index.__int__(), self)]

    @always_inline("nodebug")
    fn __getitem__(self, index: Int) -> Int:
        return self.dims[indexable["shape"](index, self)].__int__()

    @always_inline("nodebug")
    fn __setitem__(self, index: index, dimension: dim):
        alias varmut = __mlir_type[
            `!kgen.variadic<`,
            Pointer[dim, MutableAnyOrigin]._mlir_type,
            `, mut>`,
        ]
        tmp = VariadicListMem(
            UnsafePointer.address_of(self.dims).bitcast[varmut]()[]
        )
        tmp[indexable["shape"](index.__int__(), self)] = dimension

    @always_inline("nodebug")
    fn __setitem__(self, index: index, dimension: Int):
        alias varmut = __mlir_type[
            `!kgen.variadic<`,
            Pointer[dim, MutableAnyOrigin]._mlir_type,
            `, mut>`,
        ]
        tmp = VariadicListMem(
            UnsafePointer.address_of(self.dims).bitcast[varmut]()[]
        )
        tmp[indexable["shape"](index.__int__(), self)] = dim(dimension)

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

    @always_inline("nodebug")
    fn rank(self) -> index:
        """Returns the rank (number of dimensions)."""
        return len(self)

    @staticmethod
    fn unknown[rank: index]() -> Self:
        return Self()

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