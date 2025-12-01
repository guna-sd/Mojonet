from mc10.utils.time import now
from utils import StaticTuple

comptime MERSENNE_STATE_N = 624
comptime MERSENNE_STATE_M = 397
comptime MATRIX_A: UInt32 = 0x9908B0DF
comptime UMASK: UInt32 = 0x80000000
comptime LMASK: UInt32 = 0x7FFFFFFF

# The `mt19937Engine` struct provides an implementation of the Mersenne Twister MT19937
# pseudo-random number generator. Heap-allocated version for memory efficiency.


# Also planned to implement Philox counter-based random number generator... (Extremely Efficient than mt19937)...


struct MT19937State(Copyable, Movable):
    """Holds the internal state of the MT19937 generator.

    The state is stored on the heap to keep the Engine pointer small.
    """

    var seed: UInt64
    var left: Int
    var seeded: Bool
    var next: UInt32
    var state: UnsafePointer[Scalar[DType.uint32], MutOrigin.external]

    fn __init__(out self, seed: UInt64):
        """Initialize the MT19937 state with a seed."""
        self.state = alloc[UInt32](MERSENNE_STATE_N)
        self.seed = seed
        self.left = 1
        self.next = 0
        self.seeded = True
        self._init_state(seed)

    fn _init_state(mut self, seed: UInt64):
        """Initialize the state array using standard MT19937 seeding."""
        self.state[0] = (seed & 0xFFFFFFFF).cast[DType.uint32]()

        @parameter
        for i in range(1, MERSENNE_STATE_N):
            var prev = self.state[i - 1]
            # Standard MT initialization: state[i] = lowest 32 bits of
            # (1812433253 * (prev XOR (prev >> 30)) + i)
            self.state[i] = 1812433253 * (prev ^ (prev >> 30)) + i

    fn __del__(deinit self):
        """Free the heap-allocated state array."""
        if self.state:
            self.state.free()


@register_passable("trivial")
struct mt19937Engine:
    """
    A lightweight handle to the MT19937 RNG state stored on the heap.
    Copying the Engine copies the POINTER, not the state.
    This ensures all consumers share the same random stream.
    """

    var _impl: UnsafePointer[MT19937State, MutOrigin.external]

    fn __init__(out self):
        """Initialize with current time as seed."""
        self = Self(UInt64(now()))

    fn __init__(out self, seed: UInt64):
        """Initialize with a specific seed.

        Args:
            seed: The seed for the RNG.
        """
        # Allocate state on heap
        self._impl = alloc[MT19937State](1)
        self._impl.init_pointee_move(MT19937State(seed))

    @always_inline("nodebug")
    fn __init__(
        out self, state: UnsafePointer[MT19937State, MutOrigin.external]
    ):
        self._impl = state

    @always_inline("nodebug")
    fn __init__(out self, var data: MT19937State):
        """Initialize from an existing state."""
        self._impl = alloc[MT19937State](1)
        self._impl.init_pointee_move(data^)

    fn free(deinit self):
        """Explicitly free the heap-allocated state.

        Important: Call this when you're done with the Engine, or memory will leak.
        """
        if self._impl:
            self._impl.destroy_pointee()
            self._impl.free()

    @always_inline
    fn next_state(ref self):
        """Regenerate the entire state array (called every 624 generations)."""

        var ptr = self._impl
        var state = ptr[].state

        ptr[].left = MERSENNE_STATE_N
        ptr[].next = 0

        # Unrolled loop optimization usually handled by compiler,
        # but we keep logic clear.
        @parameter
        for j in range(MERSENNE_STATE_N - MERSENNE_STATE_M):
            var twisted = Self.twist(state[j], state[j + 1])
            state[j] = state[j + MERSENNE_STATE_M] ^ twisted

        @parameter
        for j in range(
            MERSENNE_STATE_N - MERSENNE_STATE_M, MERSENNE_STATE_N - 1
        ):
            var twisted = Self.twist(state[j], state[j + 1])
            state[j] = state[j + MERSENNE_STATE_N - MERSENNE_STATE_M] ^ twisted

        # Last element wrap around
        var twisted_last = Self.twist(state[MERSENNE_STATE_N - 1], state[0])
        state[MERSENNE_STATE_N - 1] = state[MERSENNE_STATE_M - 1] ^ twisted_last

    @always_inline("nodebug")
    fn __call__(ref self) -> UInt32:
        """Generate the next random UInt32."""
        if self._impl[].left == 0:
            self.next_state()

        self._impl[].left -= 1
        var y = self._impl[].state[Int(self._impl[].next)]
        self._impl[].next += 1

        # Tempering
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    # ===-------------------------------------------------------------------===#
    # SIMD Tempering - Fast parallel tempering for multiple values
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    @staticmethod
    fn temper_simd[
        width: Int
    ](vec: SIMD[DType.uint32, width]) -> SIMD[DType.uint32, width]:
        """Apply tempering to multiple UInt32 values in parallel using SIMD.

        Parameters:
            width: The SIMD vector width (e.g., 4, 8, 16 depending on hardware).

        Args:
            vec: A SIMD vector of untampered random values.

        Returns:
            A SIMD vector of tempered random values.
        """
        var y = vec
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    # ===-------------------------------------------------------------------===#
    # Batch Generation - Generate multiple random numbers at once
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn next_batch[width: Int](mut self) -> SIMD[DType.uint32, width]:
        """Generate `width` random UInt32 values in parallel.

        This method generates multiple random numbers efficiently by:
        1. Regenerating state if needed
        2. Extracting `width` consecutive state values
        3. Applying tempering in parallel using SIMD

        Parameters:
            width: The number of random values to generate (4, 8, 16, etc).

        Returns:
            A SIMD vector of `width` tempered random UInt32 values.

        Example:
            ```mojo
            from mc10.utils.mt19937 import mt19937Engine

            var rng = mt19937Engine(42)
            var randoms = rng.next_batch[8]()  # Generate 8 random numbers
            ```
        """
        var result = SIMD[DType.uint32, width]()

        # Extract `width` raw values from state
        @parameter
        for i in range(width):
            # Regenerate state if we've exhausted the current batch
            if self._impl[].left == 0:
                self.next_state()

            self._impl[].left -= 1
            result[i] = self._impl[].state[Int(self._impl[].next)]
            self._impl[].next += 1

        # Apply tempering in parallel to all values
        return self.temper_simd[width](result)

    # ===-------------------------------------------------------------------===#
    # Optimized Batch for Power-of-2 Widths (Most Hardware Support)
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn next_simd4(mut self) -> SIMD[DType.uint32, 4]:
        """Generate 4 random UInt32 values (SSE/NEON compatible)."""
        return self.next_batch[4]()

    @always_inline("nodebug")
    fn next_simd8(mut self) -> SIMD[DType.uint32, 8]:
        """Generate 8 random UInt32 values (AVX compatible)."""
        return self.next_batch[8]()

    @always_inline("nodebug")
    fn next_simd16(mut self) -> SIMD[DType.uint32, 16]:
        """Generate 16 random UInt32 values (AVX-512 compatible)."""
        return self.next_batch[16]()

    # ===-------------------------------------------------------------------===#
    # Fill Buffer - Generate random values directly into a buffer
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn fill_buffer[width: Int](mut self, mut buffer: List[UInt32]):
        """Fill a DynamicVector with random UInt32 values using batched generation.

        This is more efficient than calling __call__() repeatedly because it
        leverages SIMD for tempering multiple values in parallel.

        Parameters:
            width: The SIMD batch width to use for generation.

        Args:
            buffer: The vector to fill with random values.
        """
        var full_batches = len(buffer) // width
        var remainder = len(buffer) % width

        # Generate full SIMD batches
        for batch_idx in range(full_batches):
            var batch = self.next_batch[width]()

            @parameter
            for i in range(width):
                buffer[batch_idx * width + i] = batch[i]

        # Generate remaining values one at a time
        for i in range(remainder):
            buffer[full_batches * width + i] = self.__call__()

    # ===-------------------------------------------------------------------===#
    # Utility Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn seed(mut self, seed: UInt64):
        """Re-seed the generator (allocates new state)."""
        self.free()
        self = Self(seed)

    @always_inline("nodebug")
    fn get_seed(self) -> UInt64:
        """Get the current seed value."""
        return self._impl[].seed

    @always_inline("nodebug")
    @staticmethod
    fn mixbits(u: UInt32, v: UInt32) -> UInt32:
        """Mix upper bits of u with lower bits of v."""
        return (u & UMASK) | (v & LMASK)

    @always_inline("nodebug")
    @staticmethod
    fn twist(u: UInt32, v: UInt32) -> UInt32:
        """Apply twist transformation to two state values."""
        if (v & 1) != 0:
            return (Self.mixbits(u, v) >> 1) ^ MATRIX_A
        else:
            return Self.mixbits(u, v) >> 1

    @always_inline("nodebug")
    fn is_valid(self) -> Bool:
        """Check if the RNG state is valid."""
        if (
            self._impl[].seeded
            and 0 < self._impl[].left <= MERSENNE_STATE_N
            and self._impl[].next <= MERSENNE_STATE_N
        ):
            return True
        return False

    @always_inline("nodebug")
    fn discard(ref self, count: Int):
        """Advance the RNG by `count` steps without returning values."""
        for _ in range(0, count):
            _ = self.__call__()

    @always_inline("nodebug")
    fn save(self) -> MT19937State:
        """Save a copy of the current state (for reproducibility)."""
        return self._impl[].copy()

    @always_inline("nodebug")
    fn load(ref self, var state: MT19937State):
        """Load a previously saved state."""
        # Free old state and replace with new one
        self._impl.destroy_pointee()
        self._impl.init_pointee_move(state^)
