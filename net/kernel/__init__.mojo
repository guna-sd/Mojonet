from .constant import *
from .kernel import *
from .linalg import *
from .kutils import *
from .kmath import *
from sys.intrinsics import PrefetchOptions
alias PREFETCH_READ = PrefetchOptions().for_read().high_locality().to_data_cache()
alias PREFETCH_WRITE = PrefetchOptions().for_write().high_locality().to_data_cache()