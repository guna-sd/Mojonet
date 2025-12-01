from memory import UnsafePointer
from sys import (
    external_call,
    size_of,
    is_gpu,
    align_of,
)

alias CACHE_LINE_SIZE = 64

alias ALIGNMENT = 64  # Cache line alignment for AVX-512 / GPU coalescing
alias MIN_BLOCK_SIZE = 512  # Minimum split size to avoid fragmentation dust
