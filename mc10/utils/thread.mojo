# from memory import UnsafePointer

# struct _threadInternal:
#     var _internal: UnsafePointer[Int32]

# @explicit_destroy
# struct Thread:
#     var thread: _threadInternal

@value
struct Union[*Ts: AnyTrivialRegType]:
    alias packed = __mlir_type[
        `!kgen.pack<:!kgen.variadic<`,
        AnyTrivialRegType,
        `> `,
        Ts,
        `>`,
    ]
    alias __mlir_type = __mlir_type[
        `!pop.union<`, Self.packed,`>`
    ]

    #alias __mlir_type = __mlir_type[`!pop.union<`,Int64,`,`, Int32, `>`]
    var value: Self.__mlir_type

    fn __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))