from algorithm import vectorize, parallelize, elementwise
from time.time import perf_counter_ns, perf_counter, now

from memory import (
    UnsafePointer,
    Pointer,
    Arc,
    memcmp,
    memcpy,
    memset,
    memset_zero,
    bitcast,
)

from sys import (
    exit,
    num_physical_cores,
    external_call,
    llvm_intrinsic,
    simdwidthof,
    sizeof,
    bitwidthof,
)
from utils import (
    StaticTuple,
    IndexList,
    StringSlice,
    Writable,
    Writer,
)

from collections import Optional, OptionalReg