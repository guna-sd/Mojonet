from memory import UnsafePointer

# @register_passable("trivial")
# struct Function:
#     alias fn0 = fn() -> object
#     alias fn1 = fn(object) -> object
#     alias fn2 = fn(object, object) -> object
#     alias fn3 = fn(object, object, object) -> object
#     var _inner: UnsafePointer[NoneType]

#     fn __init__[FunctionType: AnyTrivialRegType](out self: Function, func: FunctionType):
#         var function = UnsafePointer[NoneType]().alloc(1)
#         UnsafePointer.address_of(function).bitcast[FunctionType]()[] = func
#         self._inner = function
    
#     fn __call__(owned self: Function) -> object:
#         return UnsafePointer.address_of(self._inner).bitcast[Self.fn0]()[]()
    
#     fn __call__(owned self: Function, arg1: object) -> object:
#         return UnsafePointer.address_of(self._inner).bitcast[Self.fn1]()[](arg1)
    
#     fn __call__(owned self: Function, arg1: object, arg2: object) -> object:
#         return UnsafePointer.address_of(self._inner).bitcast[Self.fn2]()[](arg1, arg2)
    
#     fn __call__(owned self: Function, arg1: object, arg2: object, arg3: object) -> object:
#         return UnsafePointer.address_of(self._inner).bitcast[Self.fn3]()[](arg1, arg2, arg3)


# @register_passable("trivial")
# struct Function[FnType: AnyTrivialRegType]:
#     var _inner: UnsafePointer[NoneType]

#     fn __init__(out self, function: FnType):
#         var func = UnsafePointer[NoneType]().alloc(1)
#         UnsafePointer.address_of(func).bitcast[FnType]()[] = function
#         self._inner = func
    
#     fn run(self) -> FnType:
#         return UnsafePointer.address_of(self._inner).bitcast[FnType]()[]

@register_passable("trivial")
struct Function[FnType: AnyTrivialRegType]:
    alias funcs = List(fn(), fn[T: AnyType](ref data: T, out result: T))
    alias func = UnsafePointer.address_of(FnType)
    var _inner: FnType

    fn __init__(out self, function: FnType):
        self._inner = function
    
    fn run(read self):
        return UnsafePointer.address_of(self._inner).bitcast[fn()]()[]()
