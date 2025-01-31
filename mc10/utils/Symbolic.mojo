from memory import ArcPointer
from utils import Variant
from mc10.__mlir import long, double, bool
from collections import Optional


@value
struct SymNodeImpl:
    var value: Variant[long, double, bool]

    fn __init__(out self):
        self.value = Variant[long, double, bool](None)

    fn __init__(out self, owned value: long):
        self.value = value

    fn __init__(out self, owned value: double):
        self.value = value

    fn __init__(out self, owned value: bool):
        self.value = value

    fn is_int(read self) -> bool:
        return self.value.isa[long]()

    fn is_float(read self) -> bool:
        return self.value.isa[double]()

    fn is_bool(read self) -> bool:
        return self.value.isa[bool]()

    fn get_bool(read self) -> Optional[bool]:
        if self.is_bool():
            return self.value.unsafe_get[bool]()
        return None

    fn get_int(read self) -> Optional[long]:
        if self.is_int():
            return self.value.unsafe_get[long]()
        return None

    fn get_float(read self) -> Optional[double]:
        if self.is_float():
            return self.value.unsafe_get[double]()
        return None


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

    fn write_to[W: Writer](self, inout writer: W):
        if self.impl[].is_float():
            writer.write(self.impl[].get_float().value())

        if self.impl[].is_int():
            writer.write(self.impl[].get_int().value())

        if self.impl[].is_bool():
            writer.write(self.impl[].get_bool().value())

    fn __repr__(self) -> String:
        return "SymNode(" + str(self) + ")"
