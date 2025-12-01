from collections import OptionalReg
from utils import Variant

# Not sure about the Implementation logic for Type (static + dynamic) got a bunch of refinement here
# TODO: make sure we have a clean DataType that is capable of Type (static + dynamic) with defaulted options...


struct DataType1[_comptime_dtype: OptionalReg[DType]]:
    comptime is_comptime: Bool = Self._comptime_dtype is not None
    alias _size = 0 if Self.is_comptime else 1
    var _runtime_value: InlineArray[DType, Self._size]

    fn __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @implicit
    fn __init__(
        out self: DataType1[_comptime_dtype = OptionalReg[DType](None)],
        var value: DType,
    ):
        self._runtime_value = type_of(self._runtime_value)(value)

    @staticmethod
    fn dtype() -> DType:
        constrained[Self.is_comptime, "dtype must be a comptime value"]()
        return Self._comptime_dtype.value()

    @always_inline
    fn dtype(self) -> DType:
        @parameter
        if Self.is_comptime:
            return Self._comptime_dtype.value()
        else:
            return self._runtime_value[0]


@register_passable("trivial")
struct DataType:
    var type: OptionalReg[DType]

    fn __init__(out self):
        self.type = None

    fn __init__(out self, dtype: DType):
        self.type = dtype

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: type_of(Bool),
    ](self) -> Bool:
        """Merge with other bools in an expression.

        Parameters:
            other_type: The type of the bool to merge with.

        Returns:
            A Bool after merging with the specified `other_type`.
        """
        return self.__bool__()

    fn __bool__(self) -> Bool:
        """Return true if the optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return self.type.__bool__()

    @always_inline("nodebug")
    fn dtype(self) -> DType:
        if self.type is not None:
            return self.type.value()
        return DType.invalid

    @always_inline("nodebug")
    fn dtype_or(self, default: DType) -> DType:
        if self.type is not None:
            return self.type.value()
        return default

    @always_inline("nodebug")
    fn dtype_or_ui8(self) -> DType:
        return self.dtype_or(DType.uint8)
