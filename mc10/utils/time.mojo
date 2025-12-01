from time.time import _monotonic_nanoseconds, _NSEC_PER_SEC


# Considered as Utility required for Allocators and Kernals for storing metadata...
# TODO: Refine for better Functionality...

@fieldwise_init
@register_passable("trivial")
struct TimeStamp(Writable):
    """TimeStamp for representing time in nanoseconds."""

    var time: UInt
    """The time in nanoseconds."""

    fn __init__(out self):
        self.time = _monotonic_nanoseconds()

    @no_inline
    fn as_nanoseconds(self) -> UInt:
        """Return the timestamp in nanoseconds."""
        return self.time

    @no_inline
    fn as_seconds(self) -> Float64:
        """Return the timestamp in seconds (fractional)."""
        return Float64(self.time) / _NSEC_PER_SEC

    @no_inline
    fn __str__(self) -> String:
        """String representation of the timestamp."""
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write the timestamp to a writer."""
        writer.write(self.as_nanoseconds(), "ns")

    fn diff(self, other: TimeStamp) -> UInt:
        """Returns the difference between two timestamps in nanoseconds."""
        return self.time - other.time


@no_inline
fn now() -> UInt:
    """Return the current time in nanoseconds."""
    return _monotonic_nanoseconds()


@no_inline
fn now_seconds() -> Float64:
    """Return the current time in seconds (fractional)."""
    return Float64(_monotonic_nanoseconds()) / _NSEC_PER_SEC


@no_inline
fn time() -> TimeStamp:
    """Return the current time as a TimeStamp object."""
    return TimeStamp()
