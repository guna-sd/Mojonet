from .constant import *
from .kernel import *
from .linalg import *
from .kutils import *
import math
from collections.optional import Optional
import time.time as time
from net.tensor import Tensor
from algorithm import vectorize, parallelize
from sys import exit, num_physical_cores
from sys.intrinsics import PrefetchOptions

alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()