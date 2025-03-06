from memory import ArcPointer
from mc10.__mlir import long, double, bool
from collections import Optional
from os import Atomic


@value
@register_passable("trivial")
struct SymNodeImpl:
    alias __mlir_type = __mlir_type[
        `!kgen.variant<`,
        long,
        `,`,
        double,
        `,`,
        bool,
        `>`,
    ]
    var value: Self.__mlir_type

    fn __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__(out self, owned value: long):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.__mlir_type, index = Int(0).value
        ](value)

    fn __init__(out self, owned value: double):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.__mlir_type, index = Int(1).value
        ](value)

    fn __init__(out self, owned value: bool):
        self.value = __mlir_op.`kgen.variant.create`[
            _type = Self.__mlir_type, index = Int(2).value
        ](value)

    fn is_int(read self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(0).value](self.value)

    fn is_float(read self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(1).value](self.value)

    fn is_bool(read self) -> Bool:
        return __mlir_op.`kgen.variant.is`[index = Int(2).value](self.value)

    fn get_int(read self) -> Optional[long]:
        if self.is_int():
            return __mlir_op.`kgen.variant.get`[index = Int(0).value](
                self.value
            )
        return None

    fn get_float(read self) -> Optional[double]:
        if self.is_float():
            return __mlir_op.`kgen.variant.get`[index = Int(1).value](
                self.value
            )
        return None

    fn get_bool(read self) -> Optional[bool]:
        if self.is_bool():
            return __mlir_op.`kgen.variant.get`[index = Int(2).value](
                self.value
            )
        return None


@value
struct SymNode:
    var impl: ArcPointer[SymNodeImpl]

    fn __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.impl)
        )

    fn is_sym(read self) -> bool:
        return True

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        if self.impl[].is_float():
            writer.write(self.impl[].get_float().value())

        if self.impl[].is_int():
            writer.write(self.impl[].get_int().value())

        if self.impl[].is_bool():
            writer.write(self.impl[].get_bool().value())

    fn __repr__(self) -> String:
        return "SymNode(" + String(self) + ")"
