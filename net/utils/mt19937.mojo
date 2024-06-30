from utils import StaticTuple
from time import now

alias MERSENNE_STATE_N = 624
alias MERSENNE_STATE_M = 397
alias MATRIX_A: UInt32 = 0x9908B0DF
alias UMASK: UInt32 = 0x80000000
alias LMASK: UInt32 = 0x7FFFFFFF

# The `mt19937Engine` struct provides an implementation of the Mersenne Twister MT19937 pseudo-random number generator in MojoNet,
# Based on 32-bit variant implemented in Pytorch.

@value
@register_passable("trivial")
struct mt19937:
    var seed: UInt64
    var left: Int
    var seeded: Bool
    var next: UInt32
    var state: StaticTuple[UInt32, MERSENNE_STATE_N]


@register_passable("trivial")
struct mt19937Engine:
    var mt19937_data: mt19937

    @always_inline("nodebug")
    fn __init__(inout self):
        self.__init__(UInt64(now()))

    @always_inline("nodebug")
    fn seed(inout self, seed: UInt64):
        self.__init__(seed)

    @always_inline("nodebug")
    fn __init__(inout self, data: mt19937):
        self.mt19937_data = data

    @always_inline("nodebug")
    fn __init__(inout self, seed: UInt64):
        var state = StaticTuple[UInt32, MERSENNE_STATE_N]()
        state[0] = seed & 0xFFFFFFFF
        for i in range(1, MERSENNE_STATE_N):
            state[i] = 1812433253 * (state[i - 1] ^ (state[i - 1] >> 30)) + i
        self.mt19937_data = mt19937(seed, 1, True, 0, state)

    @always_inline("nodebug")
    fn seed(self) -> UInt64:
        return self.mt19937_data.seed

    @always_inline("nodebug")
    @staticmethod
    fn mixbits(u: UInt32, v: UInt32) -> UInt32:
        return (u & UMASK) | (v & LMASK)

    @always_inline("nodebug")
    @staticmethod
    fn twist(u: UInt32, v: UInt32) -> UInt32:
        if (v & 1) != 0:
            return (Self.mixbits(u, v) >> 1) ^ MATRIX_A
        else:
            return Self.mixbits(u, v) >> 1

    @always_inline("nodebug")
    fn nextstate(inout self):
        var state = self.mt19937_data.state
        self.mt19937_data.left = MERSENNE_STATE_N
        self.mt19937_data.next = 0

        for j in range(MERSENNE_STATE_N - MERSENNE_STATE_M):
            var twisted = Self.twist(state[j], state[j + 1])
            state[j] = state[j + MERSENNE_STATE_M] ^ twisted

        for j in range(
            MERSENNE_STATE_N - MERSENNE_STATE_M, MERSENNE_STATE_N - 1
        ):
            var twisted = Self.twist(state[j], state[j + 1])
            state[j] = state[j + MERSENNE_STATE_N - MERSENNE_STATE_M] ^ twisted

        var twisted_last = Self.twist(state[MERSENNE_STATE_N - 1], state[0])
        state[MERSENNE_STATE_N - 1] = state[MERSENNE_STATE_M - 1] ^ twisted_last
        self.mt19937_data.state = state

    @always_inline("nodebug")
    fn __call__(inout self) -> UInt32:
        if (self.mt19937_data.left) == 0:
            self.nextstate()
        self.mt19937_data.left -= 1
        var y = self.mt19937_data.state[self.mt19937_data.next]
        self.mt19937_data.next += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    @always_inline("nodebug")
    fn is_valid(self) -> Bool:
        if (
            self.mt19937_data.seeded
            and 0 < self.mt19937_data.left <= MERSENNE_STATE_N
            and self.mt19937_data.next <= MERSENNE_STATE_N
        ):
            return True
        return False