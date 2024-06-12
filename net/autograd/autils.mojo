struct SimdFunc[T : DType, nelts : Int]:
    alias fn0 = fn(SIMD[T, nelts]) -> None
    alias fn1 = fn(SIMD[T, nelts]) -> Tensor[T]
    alias fn2 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> None
    alias fn3 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> Tensor[T]
    alias simd_args1 = 1
    alias simd_args2 = 2

struct TensorFunc[T : DType]:
    alias tfn0 =  fn(Tensor[T]) -> None
    alias tfn1 =  fn(Tensor[T]) -> Tensor[T]
    alias tfn2 =  fn(Tensor[T], Tensor[T]) -> None
    alias tfn3 =  fn(Tensor[T], Tensor[T]) -> Tensor[T]
    alias tensor_arg1 = 3
    alias tensor_arg2 = 4

struct Func[T : DType, nelts : Int]:
    alias fn0 = fn(SIMD[T, nelts]) -> None
    alias fn1 = fn(SIMD[T, nelts]) -> Tensor[T]
    alias fn2 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> None
    alias fn3 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> Tensor[T]
    alias tfn0 =  fn(Tensor[T]) -> None
    alias tfn1 =  fn(Tensor[T]) -> Tensor[T]
    alias tfn2 =  fn(Tensor[T], Tensor[T]) -> None
    alias tfn3 =  fn(Tensor[T], Tensor[T]) -> Tensor[T]

@register_passable("trivial")
struct FunctionType:

    var storage : UnsafePointer[Int32]
    """The function pointer."""

    @always_inline("nodebug")
    fn __init__[FuncType : AnyTrivialRegType](inout self, func : FuncType):
        var function = UnsafePointer[Int32]()
        UnsafePointer.address_of(function).bitcast[FuncType]()[] = func
        self.storage = function

    @always_inline("nodebug")
    fn invoke[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn0]()[]
        return func(arg0)

    @always_inline("nodebug")
    fn invokes[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn1]()[]
        return func(arg0)

    @always_inline("nodebug")
    fn invoke[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn2]()[]
        return func(arg0, arg1)

    @always_inline("nodebug")
    fn invokes[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn3]()[]
        return func(arg0, arg1)

    @always_inline("nodebug")
    fn invoke[T : DType](owned self, arg0: Tensor[T]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn0]()[]
        return func(arg0)

    @always_inline("nodebug")
    fn invokes[T : DType](owned self, arg0: Tensor[T]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn1]()[]
        return func(arg0)

    @always_inline("nodebug")
    fn invoke[T : DType](owned self, arg0: Tensor[T], arg1: Tensor[T]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn2]()[]
        return func(arg0, arg1)

    @always_inline("nodebug")
    fn invokes[T : DType](owned self, arg0: Tensor[T], arg1: Tensor[T]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn3]()[]
        return func(arg0, arg1)


@value
struct Function:
    var functions: List[FunctionType]

    @always_inline("nodebug")
    fn __init__(inout self):
        self.functions = List[FunctionType]()

    fn __init__[FuncType : AnyTrivialRegType](inout self, func: FuncType):
        self.functions = List[FunctionType]()
        var func_type = FunctionType(func)
        self.functions.append(func_type)
    
    fn __getitem__(self, index : Int) -> FunctionType:
        return self.functions[index]
    
    fn __setitem__(inout self, index : Int, value : FunctionType):
        self.functions[index] = value

    @always_inline("nodebug")
    fn append[FuncType : AnyTrivialRegType](inout self, func: FuncType):
        self.functions.append(FunctionType(func))

    @always_inline("nodebug")
    fn __call__[T : DType, nelts : Int = simdwidthof[T]()](self, arg0: SIMD[T, nelts]):
        for func_type in self.functions:
            func_type[].invoke(arg0)

    @always_inline("nodebug")
    fn invokes_all[T : DType, nelts : Int = simdwidthof[T]()](self, arg0: SIMD[T, nelts]) -> List[Tensor[T]]:
        var results: List[Tensor[T]] = List[Tensor[T]]()
        for func_type in self.functions:
            var result = func_type[].invokes(arg0)
            results.append(result)
        return results

    @always_inline("nodebug")
    fn __call__[T : DType, nelts : Int = simdwidthof[T]()](self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]):
        for func_type in self.functions:
            func_type[].invoke(arg0, arg1)

    @always_inline("nodebug")
    fn invokes_all[T : DType, nelts : Int = simdwidthof[T]()](self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]) -> List[Tensor[T]]:
        var results: List[Tensor[T]] = List[Tensor[T]]()
        for func_type in self.functions:
            var result = func_type[].invokes(arg0, arg1)
            results.append(result)
        return results

    @always_inline("nodebug")
    fn __call__[T : DType](self, arg0: Tensor[T]):
        for func_type in self.functions:
            func_type[].invoke(arg0)

    @always_inline("nodebug")
    fn invokes_all[T : DType](self, arg0: Tensor[T]) -> List[Tensor[T]]:
        var results: List[Tensor[T]] = List[Tensor[T]]()
        for func_type in self.functions:
            var result = func_type[].invokes(arg0)
            results.append(result)
        return results

    @always_inline("nodebug")
    fn __call__[T : DType](self, arg0: Tensor[T], arg1: Tensor[T]):
        for func_type in self.functions:
            func_type[].invoke(arg0, arg1)

    @always_inline("nodebug")
    fn invokes_all[T : DType](self, arg0: Tensor[T], arg1: Tensor[T]) -> List[Tensor[T]]:
        var results: List[Tensor[T]] = List[Tensor[T]]()
        for func_type in self.functions:
            var result = func_type[].invokes(arg0, arg1)
            results.append(result)
        return results

@value
struct Operation[T : DType]:
    """
    Represents an operation performed on tensors.
    """
    var name: String
    var forward: List[Function]
    var backward: List[Function]
    ...

@value
struct Node[T : DType]:
    """
    Node in the computation graph.
    """
    var operation: Operation[T]
    var inputs: List[Node[T]]
    var outputs: List[Node[T]]
    var grad_fn: List[Function]