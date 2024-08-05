@value
struct Parameter[type: DType = DType.float32](CollectionElement):
    fn __call__(self, *shapes: Int) -> Tensor[type]:
        return Tensor[type](shapes, requires_grad=True).random()


@always_inline("nodebug")
fn parameter[type: DType = DType.float32](*shapes: Int) -> Tensor[type]:
    return Tensor[type](shapes, requires_grad=True).random()
