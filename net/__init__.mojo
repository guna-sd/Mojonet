from utils.variant import Variant
from collections import Optional
import time.time as time
import math
from algorithm import vectorize, parallelize, elementwise
from sys import exit, num_physical_cores
from .tensor import *
from .nn import *
from .autograd import *
from .kernel import *
from .checkpoint import *