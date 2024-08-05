# The `randn` struct implements a pseudo-random number generator in MojoNet, using both a simple linear congruential generator (LCG)
# and a variant of the xorshift64* algorithm for generating random numbers of various types. This generator is designed
# primarily for use within the MojoNet framework and is not intended for general-purpose use.

# This struct is designed for simplicity and performance in scenarios where high statistical quality is
# not critical. For production use or applications requiring higher randomness quality, consider using MersenneTwister.

import time


struct randn:
    var _seed: Int

    @always_inline("nodebug")
    fn __init__(inout self):
        self._seed = time.now()

    @always_inline("nodebug")
    fn __init__(inout self, seed: Int):
        self._seed = seed

    @always_inline("nodebug")
    fn seed(inout self):
        self._seed = time.now()

    @always_inline("nodebug")
    fn seed(inout self, seed: Int):
        self._seed = seed

    @always_inline("nodebug")
    fn lcg(inout self) -> UInt64:
        self._seed = (self._seed * 1103515245 + 12345) & 2147483647
        return UInt64(self._seed)

    @staticmethod
    @always_inline("nodebug")
    fn u64(inout state: UInt64) -> UInt64:
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        return ((state * 0x2545F4914F6CDD1D)).cast[DType.uint64]()

    @always_inline("nodebug")
    fn randint8(inout self) -> Int8:
        var val = UInt64(self.lcg())
        return Int8((self.u64(val)).cast[DType.int8]()) % Int8.MAX_FINITE

    @always_inline("nodebug")
    fn randuint8(inout self) -> UInt8:
        var val = UInt64(self.lcg())
        return UInt8((self.u64(val)).cast[DType.uint8]()) % UInt8.MAX_FINITE

    @always_inline("nodebug")
    fn randint16(inout self) -> Int16:
        var val = UInt64(self.lcg())
        return Int16((self.u64(val)).cast[DType.int16]()) % Int16.MAX_FINITE

    @always_inline("nodebug")
    fn randuint16(inout self) -> UInt16:
        var val = UInt64(self.lcg())
        return UInt16((self.u64(val)).cast[DType.uint16]()) % UInt16.MAX_FINITE

    @always_inline("nodebug")
    fn randint32(inout self) -> Int32:
        var val = UInt64(self.lcg())
        return Int32((self.u64(val)).cast[DType.int32]()) % Int32.MAX_FINITE

    @always_inline("nodebug")
    fn randuint32(inout self) -> UInt32:
        var val = UInt64(self.lcg())
        return UInt32((self.u64(val)).cast[DType.uint32]()) % UInt32.MAX_FINITE

    @always_inline("nodebug")
    fn randint64(inout self) -> Int64:
        var val = UInt64(self.lcg())
        return Int64((self.u64(val)).cast[DType.int64]()) % Int64.MAX_FINITE

    @always_inline("nodebug")
    fn randuint64(inout self) -> UInt64:
        var val = UInt64(self.lcg())
        return UInt64((self.u64(val)).cast[DType.uint64]()) % UInt64.MAX_FINITE

    @always_inline("nodebug")
    fn randf16(inout self) -> Float16:
        return Float16(
            (self.randint16()).cast[DType.float16]() / Float16.MAX_FINITE
        )

    @always_inline("nodebug")
    fn randf32(inout self) -> Float32:
        return Float32(
            (self.randint32()).cast[DType.float32]() / Float32.MAX_FINITE * 1e29
        )

    @always_inline("nodebug")
    fn randf64(inout self) -> Float64:
        return Float64(
            (self.randint64()).cast[DType.float64]()
            / Float64.MAX_FINITE
            * 1e289
        )

    @always_inline("nodebug")
    fn randbf16(inout self) -> BFloat16:
        return BFloat16(
            (self.randint64()).cast[DType.bfloat16]()
            / BFloat16.MAX_FINITE
            * 1e19
        )


@always_inline("nodebug")
fn rand_n[type: DType](ptr: DTypePointer[type], count: Int):
    var rand = randn()

    @parameter
    if type.is_int8():
        for i in range(count):
            ptr[i] = rand.randint8().cast[type]()

    @parameter
    if type.is_uint8():
        for i in range(count):
            ptr[i] = rand.randuint8().cast[type]()

    @parameter
    if type.is_int16():
        for i in range(count):
            ptr[i] = rand.randint16().cast[type]()

    @parameter
    if type.is_uint16():
        for i in range(count):
            ptr[i] = rand.randuint16().cast[type]()

    @parameter
    if type.is_int32():
        for i in range(count):
            ptr[i] = rand.randint32().cast[type]()

    @parameter
    if type.is_uint32():
        for i in range(count):
            ptr[i] = rand.randuint32().cast[type]()

    @parameter
    if type.is_int64():
        for i in range(count):
            ptr[i] = rand.randint64().cast[type]()

    @parameter
    if type.is_uint64():
        for i in range(count):
            ptr[i] = rand.randuint64().cast[type]()

    @parameter
    if type.is_float16():
        for i in range(count):
            ptr[i] = rand.randf16().cast[type]()

    @parameter
    if type.is_bfloat16():
        for i in range(count):
            ptr[i] = rand.randbf16().cast[type]()

    @parameter
    if type.is_float32():
        for i in range(count):
            ptr[i] = rand.randf32().cast[type]()

    @parameter
    if type.is_float64():
        for i in range(count):
            ptr[i] = rand.randf64().cast[type]()


@always_inline("nodebug")
fn rand_n[type: DType](ptr: DTypePointer[type], count: Int, seed: Int):
    var rand = randn(seed)

    @parameter
    if type.is_int8():
        for i in range(count):
            ptr[i] = rand.randint8().cast[type]()

    @parameter
    if type.is_uint8():
        for i in range(count):
            ptr[i] = rand.randuint8().cast[type]()

    @parameter
    if type.is_int16():
        for i in range(count):
            ptr[i] = rand.randint16().cast[type]()

    @parameter
    if type.is_uint16():
        for i in range(count):
            ptr[i] = rand.randuint16().cast[type]()

    @parameter
    if type.is_int32():
        for i in range(count):
            ptr[i] = rand.randint32().cast[type]()

    @parameter
    if type.is_uint32():
        for i in range(count):
            ptr[i] = rand.randuint32().cast[type]()

    @parameter
    if type.is_int64():
        for i in range(count):
            ptr[i] = rand.randint64().cast[type]()

    @parameter
    if type.is_uint64():
        for i in range(count):
            ptr[i] = rand.randuint64().cast[type]()

    @parameter
    if type.is_float16():
        for i in range(count):
            ptr[i] = rand.randf16().cast[type]()

    @parameter
    if type.is_bfloat16():
        for i in range(count):
            ptr[i] = rand.randbf16().cast[type]()

    @parameter
    if type.is_float32():
        for i in range(count):
            ptr[i] = rand.randf32().cast[type]()

    @parameter
    if type.is_float64():
        for i in range(count):
            ptr[i] = rand.randf64().cast[type]()
