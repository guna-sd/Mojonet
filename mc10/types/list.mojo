"""
A custom implementation of the stdlib List.
"""

from mc10.utils.debuggable import abort
from mc10.types.Array import indexable
from memory import UnsafePointer


struct Node[
    ElementType: Copyable & Movable,
]:
    """A node in a linked list data structure.

    Parameters:
        ElementType: The type of element stored in the node.
    """

    alias _NodePointer = UnsafePointer[Self, MutOrigin.external]

    var value: Self.ElementType
    """The value stored in this node."""
    var prev: Self._NodePointer
    """The previous node in the list."""
    var next: Self._NodePointer
    """The next node in the list."""

    fn __init__(
        out self,
        var value: Self.ElementType,
        prev: Optional[Self._NodePointer],
        next: Optional[Self._NodePointer],
    ):
        """Initialize a new Node with the given value and optional prev/next
        pointers.

        Args:
            value: The value to store in this node.
            prev: Optional pointer to the previous node.
            next: Optional pointer to the next node.
        """
        self.value = value^
        self.prev = prev.value() if prev else Self._NodePointer()
        self.next = next.value() if next else Self._NodePointer()

    fn __str__[
        Element: Writable & Copyable & Movable
    ](self: Node[Element]) -> String:
        """Convert this node's value to a string representation.

        Parameters:
            Element: Used to conditionally enable this function if
              `ElementType` is `Writable`.

        Returns:
            String representation of the node's value.
        """
        return String.write(self.value)

    @no_inline
    fn write_to[
        Element: Writable & Copyable & Movable, W: Writer
    ](self: Node[Element], mut writer: W):
        """Write this node's value to the given writer.

        Parameters:
            Element: Used to conditionally enable this function if
              `ElementType` is `Writable`.
            W: The type of writer to write the value to.

        Args:
            writer: The writer to write the value to.
        """
        writer.write(self.value)


## TODO: Yet to Implement List Planned for next release
