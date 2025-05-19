alias kstrided = LayoutType.Strided
alias ksparse = LayoutType.Sparse
alias kSparseCsr = LayoutType.SparseCsr
alias kMkldnn = LayoutType.Mkldnn


@fieldwise_init
@register_passable("trivial")
struct LayoutType(AnyType, Copyable, EqualityComparable, Hashable, KeyElement, Representable, Stringable, Writable):
    alias Strided = LayoutType(0)
    alias Sparse = LayoutType(1)
    alias SparseCsr = LayoutType(2)
    alias Mkldnn = LayoutType(3)
    var value: Int8

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

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the LayoutType.

        Returns:
            An integer hash value based on the LayoutType's value.
        """
        return hash(UInt8(self.value.cast[DType.uint8]()))

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
        if self == LayoutType.SparseCsr:
            return writer.write("SparseCsr")
        if self == LayoutType.Mkldnn:
            return writer.write("Mkldnn")
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
    @staticmethod
    fn is_valid(value: Int8) -> Bool:
        return value >= 0 and value <= 3
