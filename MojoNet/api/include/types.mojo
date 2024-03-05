from random import rand
from memory import memset
from memory.buffer import Buffer
from utils.index import StaticIntTuple, Index
from utils.list import Dim, DimList
from collections.vector import InlinedFixedVector, DynamicVector
from runtime.llcl import num_physical_cores, Runtime
from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
import math



struct Tensor[dtype : DType]:
  var data : DTypePointer[dtype]
  var dim : DTypePointer[dtype]
  var size : Int
  alias simd_width = simdwidthof[dtype]()