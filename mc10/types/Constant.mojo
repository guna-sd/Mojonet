##TODO: deprecated find a new way to store literals

# @value
# @register_passable("trivial")
# struct Constant:
#     """Constant representing an Int or Float literal."""

#     alias _mlir_type = __mlir_type[
#         `!kgen.variant<`,
#         __mlir_type.`!kgen.int_literal`,
#         `, `,
#         __mlir_type.`!kgen.float_literal`,
#         `>`,
#     ]
#     var value: Self._mlir_type

#     fn __init__(out self):
#         self = Self(0)

#     @implicit
#     fn __init__(out self, value: IntLiteral):
#         self.value = __mlir_op.`kgen.variant.create`[
#             _type = Self._mlir_type, index = Int(0).value
#         ](value.value)

#     @implicit
#     fn __init__(out self, value: FloatLiteral):
#         self.value = __mlir_op.`kgen.variant.create`[
#             _type = Self._mlir_type, index = Int(1).value
#         ](value.value)

#     @always_inline("nodebug")
#     fn __bool__(self) -> Bool:
#         var flval: Bool = __mlir_op.`kgen.variant.is`[index = Int(1).value](
#             self.value
#         )
#         if flval:
#             return FloatLiteral(
#                 __mlir_op.`kgen.variant.get`[index = Int(1).value](self.value)
#             ).__bool__()

#         return IntLiteral(
#             __mlir_op.`kgen.variant.get`[index = Int(0).value](self.value)
#         ).__bool__()

#     @always_inline("nodebug")
#     fn __as_bool__(self) -> Bool:
#         return self.__bool__()

#     fn __is_float__(self) -> Bool:
#         return __mlir_op.`kgen.variant.is`[index = Int(1).value](self.value)

#     @always_inline("nodebug")
#     fn __int__(self) -> Int:
#         return Int(self.__index__())

#     @always_inline("nodebug")
#     fn __float__(self) -> Float64:
#         return Float64(self.__float_literal__())

#     @always_inline("nodebug")
#     fn __index__(self) -> __mlir_type.index:
#         return __mlir_op.`kgen.int_literal.convert`[_type = __mlir_type.index](
#             self.__int_literal__().value
#         )

#     @always_inline("nodebug")
#     fn __int_literal__(self) -> IntLiteral:
#         var flval: Bool = __mlir_op.`kgen.variant.is`[index = Int(1).value](
#             self.value
#         )
#         if flval:
#             return FloatLiteral(
#                 __mlir_op.`kgen.variant.get`[index = Int(1).value](self.value)
#             ).__int_literal__()

#         return IntLiteral(
#             __mlir_op.`kgen.variant.get`[index = Int(0).value](self.value)
#         )

#     @always_inline("nodebug")
#     fn __float_literal__(self) -> FloatLiteral:
#         var flval: Bool = __mlir_op.`kgen.variant.is`[index = Int(1).value](
#             self.value
#         )
#         if flval:
#             return FloatLiteral(
#                 __mlir_op.`kgen.variant.get`[index = Int(1).value](self.value)
#             )

#         return FloatLiteral(
#             __mlir_op.`kgen.variant.get`[index = Int(0).value](self.value)
#         )

#     @always_inline("nodebug")
#     fn __str__(self) -> String:
#         return String.write(self)

#     @always_inline("nodebug")
#     fn __repr__(self) -> String:
#         return String("constant(", self, ")")
    
#     fn write_to[W: Writer](self, mut string: W):
#         if self.__is_float__():
#             string.write(self.__float__())
#         else:
#             string.write(self.__int__())

## TODO: Still a workaround not sure if this is the right way to do it.
@value
@register_passable("trivial")
struct OptionalParamInt[dim_parametric: Int = -1342]:
    """A class to represent an optionally parametric Int.
    If dim_parametric is known, the get method can be evaluated at compile time.
    Otherwise, the get method will be evaluated at runtime using the dynamic
    value supplied to the constructor.

    Parameters:
        dim_parametric: The optional Int parameter.
    """

    var dim_dynamic: Int

    @always_inline("nodebug")
    fn __init__(out self, dim_dynamic: Int):
        self.dim_dynamic = dim_dynamic

    # @always_inline("nodebug")
    # fn get(self) -> Int:
    #     @parameter
    #     if dim_parametric.has_value()-:
    #         return dim_parametric.get()
    #     else:
    #         return self.dim_dynamic

    # @always_inline("nodebug")
    # fn __eq__(self, rhs: OptionalParamInt[dim_parametric]) -> Bool:
    #     return self.get() == rhs.get()

    # @always_inline("nodebug")
    # fn __ne__(self, rhs: OptionalParamInt[dim_parametric]) -> Bool:
    #     return self.get() != rhs.get()