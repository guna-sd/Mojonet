trait Operatable:
    fn forward(self):
        ...
    fn backward(self):
        ...

struct Valued[T : DType]:
    alias Type = Variant[Tensor[T], Scalar[T], SIMD[T]]

    var value : Self.Type

    fn __init__(inout self, value : Tensor[T]):
        self.value = value
    
    fn __init__(inout self, value : Scalar[T]):
        self.value = value

    fn __init__(inout self, value : SIMD[T]):
        self.value = value
    
    fn get_tensor(self) -> Tensor[T]:
        if not self.value.isa[Tensor[T]]():
            print("doesn't contain Tensor could be owner of a Scalar or a SIMD object")
            return Tensor[T]()
        return self.value[Tensor[T]]
    
    fn get_scalar(self) -> Scalar[T]:
        if not self.value.isa[Scalar[T]]():
            print("doesn't contain Scalar could be owner of Tensor or a SIMD object")
            return Scalar[T]()
        return self.value[Scalar[T]]
    
    fn get_simd(self) -> SIMD[T]:
        if not self.value.isa[SIMD[T]]():
            print("doesn't contain a SIMD value could be a owner of a Tensor or a Scalar")
            return SIMD[T]()
        return self.value[SIMD[T]]