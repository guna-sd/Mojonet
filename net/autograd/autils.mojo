struct SimdFunc[T : DType, nelts : Int]:
    alias fn0 = fn(SIMD[T, nelts]) -> None
    alias fn1 = fn(SIMD[T, nelts]) -> Tensor[T]
    alias fn2 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> None
    alias fn3 = fn(SIMD[T, nelts], SIMD[T, nelts]) -> Tensor[T]

struct TensorFunc[T : DType]:
    alias tfn0 =  fn(Tensor[T]) -> None
    alias tfn1 =  fn(Tensor[T]) -> Tensor[T]
    alias tfn2 =  fn(Tensor[T], Tensor[T]) -> None
    alias tfn3 =  fn(Tensor[T], Tensor[T]) -> Tensor[T]

@register_passable("trivial")
struct Function:

    var storage : UnsafePointer[Int32]
    """The function pointer."""

    @always_inline
    fn __init__[FuncType : AnyTrivialRegType](inout self, func : FuncType):
        var function = UnsafePointer[Int32]()
        UnsafePointer.address_of(function).bitcast[FuncType]()[] = func
        self.storage = function

    @always_inline
    fn invoke[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn0]()[]
        return func(arg0)

    @always_inline
    fn invokes[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn1]()[]
        return func(arg0)

    @always_inline
    fn invoke[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn2]()[]
        return func(arg0, arg1)

    @always_inline
    fn invokes[T : DType, nelts : Int = simdwidthof[T]()](owned self, arg0: SIMD[T, nelts], arg1: SIMD[T, nelts]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[SimdFunc[T, nelts].fn3]()[]
        return func(arg0, arg1)

    @always_inline
    fn invoke[T : DType](owned self, arg0: Tensor[T]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn0]()[]
        return func(arg0)

    @always_inline
    fn invokes[T : DType](owned self, arg0: Tensor[T]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn1]()[]
        return func(arg0)

    @always_inline
    fn invoke[T : DType](owned self, arg0: Tensor[T], arg1: Tensor[T]) -> None:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn2]()[]
        return func(arg0, arg1)

    @always_inline
    fn invokes[T : DType](owned self, arg0: Tensor[T], arg1: Tensor[T]) -> Tensor[T]:
        var func = UnsafePointer.address_of(self.storage).bitcast[TensorFunc[T].tfn3]()[]
        return func(arg0, arg1)

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