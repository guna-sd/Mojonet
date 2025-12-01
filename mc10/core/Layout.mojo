from hashlib import Hasher


# This is a small workaround for Layout Tag which tells the framwork to choose a way of memory representation for underlying data...


@fieldwise_init
@register_passable("trivial")
struct LayoutType(
    AnyType,
    Copyable,
    Equatable,
    Hashable,
    KeyElement,
    Representable,
    Stringable,
    Writable,
):
    alias Strided = LayoutType(0)
    alias Sparse = LayoutType(1)
    var value: Int8

    fn __init__(out self):
        self = Self.default()

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the LayoutType.

        Returns:
            The name of the layout.
        """

        return String.write(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the LayoutType e.g. `"LayoutType.Strided"`.

        Returns:
            The representation of the layout.
        """
        return "LayoutType." + String(self)

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self.value)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this layout to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        if self == LayoutType.Strided:
            return writer.write("Strided")
        if self == LayoutType.Sparse:
            return writer.write("Sparse")
        return writer.write("Unknown layout")

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares one LayoutType to another for equality.

        Args:
            rhs: The LayoutType to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compares one LayoutType to another for inequality.

        Args:
            rhs: The LayoutType to compare against.

        Returns:
            False if the Layouts are the same and True otherwise.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        """Compares one LayoutType to another for equality.

        Args:
            rhs: The LayoutType to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        """Compares one LayoutType to another for inequality.

        Args:
            rhs: The LayoutType to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self != rhs

    @always_inline("nodebug")
    fn is_strided(self) -> Bool:
        return self == LayoutType.Strided

    @always_inline("nodebug")
    fn is_sparse(self) -> Bool:
        return self == LayoutType.Sparse

    @staticmethod
    fn default() -> LayoutType:
        return LayoutType.Strided
