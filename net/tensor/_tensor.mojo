@value
struct Tensor[Type: DType = DType.float32](CollectionElement):
    var data: StorageImpl[Type]
    var shape: Arc[shape]
    var dtype: DType
    var device: Device
    var requires_grad: Bool
    var grad: Optional[Arc[Tensor[Type]]]
    var is_leaf: Bool
    var id: Int

    fn __init__(inout self):
        self.data = StorageImpl[Type]()
        self.shape = shape()
        self.dtype = Type
        self.device = Device.CPU
        self.requires_grad = False
        self.grad = None
        self.is_leaf = True
        self.id = unique_id()
    
    fn __init__(inout self, shapes: List[Int], device: Device = Device.CPU, requires_grad: Bool = False):
        self.data = StorageImpl[Type]()
        self.shape = shape(shapes)
        self.dtype = Type
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = True
        self.id = unique_id()
        if self.requires_grad:
            self.grad = Arc(Tensor[Type]())

    fn __init__(inout self, *shapes: Int, device: Device = Device.CPU, requires_grad: Bool = False):
        self.data = StorageImpl[Type]()
        self.shape = shape(shapes)
        self.dtype = Type
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = True
        self.id = unique_id()
        if self.requires_grad:
            self.grad = Arc(Tensor[Type]())

    @always_inline("nodebug")
    fn format_to(self, inout writer: Formatter):
        TensorPrinter(self.data.raw_ptr[], self.shape[].shapes, writer)
    
    @staticmethod
    fn cast[dtype: DType](tensor) -> Tensor[dtype] as casted:
        casted = Tensor[dtype]()

fn unique_id() -> Int as id:
    time_stamp = perf_counter_ns()
    generator = randn(time_stamp^2)
    random_number = generator.randint64()
    id = int(time_stamp ^ random_number)

# trait callable:
#     def __init__(inout self, inputs: List[Tensor]):
#         ...
#     fn forward(self, *args: Tensor) -> Tensor:
#         ...
#     fn backward(self, grad_output: Tensor) -> List[Tensor]:
#         ...
#     fn save_for_backward(self, *tensors: Tensor):
#         ...
    
# struct Function:
#     var inputs: List[Tensor]
#     var output: Optional[Tensor]
#     var saved_tensors: List[Tensor]
