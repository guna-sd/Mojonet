from collections import Optional, InlinedFixedVector
from utils.variant import Variant
from utils.index import IndexList

@value
struct __ShapeElementType:
    alias static = __ShapeElementType(0)
    alias dynamic = __ShapeElementType(1)
    var value: Int

    fn __eq__(self, other: Self) -> Bool:
        return self.value.__eq__(other.value)

    fn __ne__(self, other: Self) -> Bool:
        return self.value.__ne__(other.value)
    
    fn isdynamic(self) -> Bool:
        return self.value == Self.dynamic.value
    
    fn isstatic(self) -> Bool:
        return self.value == Self.static.value

    fn __str__(self) -> String:
        if self.isstatic():
            return "Static"
        elif self.isdynamic():
            return "Dynamic"
        else:
            return "Unknown"
    
    fn __is__(self, other: Self) -> Bool:
        return self == other
    
    fn __isnot__(self, other: Self) -> Bool:
        return self != other 

@value
struct ShapeElement(AnyType, CollectionElement, EqualityComparable):
    var elmtype: __ShapeElementType
    var name: String
    var static: Optional[Int]

    fn __init__[T: Intable](out self, static: T):
        self.elmtype = __ShapeElementType.static
        self.name = "static_shape_element"
        self.static = int(static)
    
    fn __init__[T: Stringable](out self, owned name: T):
        self.elmtype = __ShapeElementType.dynamic
        self.name = str(name)
        self.static = None

    fn __int__(self) -> Int:
        return self.static.or_else(0)
    
    fn __eq__(self, other: Self) -> Bool:
        return self.element() == other.element()
    
    fn __ne__(self, other: Self) -> Bool:
        return self.element() != other.element()

    fn isdynamic(self) -> Bool:
        return self.elmtype.isdynamic()
    
    fn isstatic(self) -> Bool:
        return self.elmtype.isstatic()
    
    fn element(self) -> Int:
        "Returns the static shape element, 0 if not a static shape element."
        return self.static.or_else(0)


struct shape:
    alias __staticrank = 24
    var __type: __ShapeElementType
    var rank: UInt
    var dynamic: List[Int, True]
    var static: IndexList[size=Self.__staticrank]

    fn __init__(out self, rank: Int):
        self.__type = __ShapeElementType.static if rank <= Self.__staticrank else __ShapeElementType.dynamic
        self.dynamic = List[Int, True]()
        if self.__type.isdynamic():
            self.dynamic = List[Int, True](capacity=rank)
        self.static = IndexList[size=Self.__staticrank]()
        self.rank = rank
