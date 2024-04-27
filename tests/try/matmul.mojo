from random import rand
from memory import memset_zero
from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from math import abs

alias type = DType.float32
alias M = 1024  # M of A and C
alias N = 1024  # N of B and C
alias K = 1024  # N of A and M of B
# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2
alias tile_n = 64  # N must be a multiple of this
alias tile_k = 4  # K must be a multiple of this

struct Matrix[M: Int, N: Int]:
    var data: DTypePointer[type]

    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(M * N)
        memset_zero(self.data, M * N)

    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(M * N)
        rand(data, M * N)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[type, 1]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[type, 1]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.simd_load[nelts](y * self.N + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.simd_store[nelts](y * self.N + x, val)


fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.M):
        for k in range(A.N):
            for n in range(C.N):
                C[m, n] += A[m, k] * B[k, n]


fn matmul_vectorized(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.M):
        for k in range(A.N):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.N]()

fn matmul_parallelized(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.N):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.N]()

    parallelize[calc_row](C.M, C.M)

# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul
fn matmul_tiled(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[dot, nelts, size=tile_x]()

        tile[calc_tile, tile_n, tile_k](C.N, B.M)

    parallelize[calc_row](C.M, C.M)


# Unroll the vectorized loop by a constant factor
fn matmul_unrolled(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            @unroll(tile_y)
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, tile_x, unroll_factor]()

        tile[calc_tile, tile_n, tile_k](C.N, B.M)

    parallelize[calc_row](C.M, C.M)

# Unroll the vectorized loop by a constant factor.
fn matmul_tiled_unrolled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(m, n + x, C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, tile_x, unroll_factor]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.N, C.N)

    parallelize[calc_row](C.M, C.M)


@always_inline
fn test_matrix_equal[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](inout C: Matrix, A: Matrix, B: Matrix) raises -> Bool:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    var result = Matrix[M, N]()
    _ = func(result, A, B)
    for i in range(C.M):
        for j in range(C.N):
            if C[i, j] != result[i, j]:
                return False
    return True


fn test_all() raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    matmul_naive(C, A, B)

    if not test_matrix_equal[matmul_vectorized](C, A, B):
        raise Error("Vectorize output does not match naive implementation")
    if not test_matrix_equal[matmul_parallelized](C, A, B):
        raise Error("Parallelize output does not match naive implementation")
    if not test_matrix_equal[matmul_tiled](C, A, B):
        raise Error("Tiled output does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled](C, A, B):
        raise Error("Unroll output does not match naive implementation")

    A.data.free()
    B.data.free()
    C.data.free()

var MAX_ITERS = 1000

def mandelbrot_kernel(c): 
  z = c
  nv = 0
  for i in range(MAX_ITERS):
    if abs(z) > 2:
      break
    z = z*z + c
    nv += 1
  return nv

fn main() raises:
    # var C = Matrix[M, N]()
    # var A = Matrix[M, K].rand()
    # var B = Matrix[K, N].rand()
    # matmul_tiled_unrolled_parallelized(C, A, B)
    
    print(mandelbrot_kernel(5))

import time
from collections.optional import Optional

@value
struct random:
    var _seed : Optional[Int]

    fn seed(inout self):
        """
        Seeds the random number generator using the current time.
        """
        self._seed = time.now()
    
    fn seed(inout self, seed : Int):
        """
        Seeds the random number generator with a specified seed.

        Args:
            seed: The seed value to use for the random number generator.
        """
        self._seed = seed

    fn uniform_float(self) -> Float:
        """
        Generates a random float between 0 and 1.
        """
        if self._seed.has_value:
            # Use the seed value if provided
            srand(self._seed.get)
        return rand()

    fn uniform_int(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer in the range [low, high].
        """
        if self._seed:
            # Use the seed value if provided
            srand(self._seed.get)
        return randint(low, high)

    fn normal(self, mean: Float, std_dev: Float) -> Float:
        """
        Generates a random float from a normal distribution with the given mean and standard deviation.
        """
        if self._seed.has_value:
            # Use the seed value if provided
            srand(self._seed.get)
        return mean + std_dev * randn()

    fn randn(self) -> Float:
        """
        Generates a random float from a standard normal distribution (mean=0, std_dev=1).
        """
        return self.normal(0.0, 1.0)

    fn randint(self, low: Int, high: Int) -> Int:
        """
        Generates a random integer in the range [low, high].
        """
        return self.uniform_int(low, high)

    fn choice(self, choices: List[T]) -> T:
        """
        Chooses a random element from the given list of choices.
        """
        if self._seed.has_value:
            # Use the seed value if provided
            srand(self._seed.get)
        var index = randint(0, len(choices) - 1)
        return choices[index]

    fn shuffle(self, array: List[T]) -> List[T]:
        """
        Shuffles the elements of the given array in-place and returns the shuffled array.
        """
        if self._seed.has_value:
            # Use the seed value if provided
            srand(self._seed.get)
        var shuffled = array.clone()
        shuffled.shuffle()
        return shuffled
