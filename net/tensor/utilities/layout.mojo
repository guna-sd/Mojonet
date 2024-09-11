alias kstrided = Layout.Strided
alias ksparse = Layout.Sparse
alias kSparseCsr = Layout.SparseCsr
alias kMkldnn = Layout.Mkldnn


@value
struct Layout(Stringable, Formattable, Representable, KeyElement):
    alias Strided = Layout(0)
    alias Sparse = Layout(1)
    alias SparseCsr = Layout(2)
    alias Mkldnn = Layout(3)
    var value: Int8

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the Layout.

        Returns:
            The name of the layout.
        """

        return String.format_sequence(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the Layout e.g. `"Layout.Strided"`.

        Returns:
            The representation of the layout.
        """
        return "Layout." + str(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the Layout.

        Returns:
            An integer hash value based on the Layout's value.
        """
        return hash(UInt8(self.value.cast[DType.uint8]()))

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this layout to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        if self == Layout.Strided:
            return writer.write_str("Strided")
        if self == Layout.Sparse:
            return writer.write_str("Sparse")
        if self == Layout.SparseCsr:
            return writer.write_str("SparseCsr")
        if self == Layout.Mkldnn:
            return writer.write_str("Mkldnn")
        return writer.write_str("Unknown layout")

    @always_inline("nodebug")
    fn __eq__(self, rhs: Layout) -> Bool:
        """Compares one Layout to another for equality.

        Args:
            rhs: The Layout to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Layout) -> Bool:
        """Compares one Layout to another for inequality.

        Args:
            rhs: The Layout to compare against.

        Returns:
            False if the Layouts are the same and True otherwise.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Layout) -> Bool:
        """Compares one Layout to another for equality.

        Args:
            rhs: The Layout to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self == rhs


    @always_inline("nodebug")
    fn __isnot__(self, rhs: Layout) -> Bool:
        """Compares one Layout to another for inequality.

        Args:
            rhs: The Layout to compare against.

        Returns:
            True if the Layouts are the same and False otherwise.
        """
        return self != rhs