struct Serialize:
    alias MAGIC_NUMBER: UInt64 = 0xFFFFFFFFFFFFFFFF
    var storage: InlinedFixedVector[Bytes]
    var shapes: List[Int]

    fn __init__(
        inout self, shapes: List[Int], storage: InlinedFixedVector[Bytes]
    ):
        self.storage = storage
        self.shapes = shapes

    fn __init__[type: DType](inout self, tensor: Tensor[type]):
        self.shapes = tensor.shapes().Shapes()
        self.storage = InlinedFixedVector[Bytes](
            capacity=(tensor.num_elements() + self.shapes.__len__() + 2)
        )
        self.fromtensor(tensor)

    fn fromtensor[type: DType](inout self, tensor: Tensor[type]):
        var shapes = tensor.shapes().Shapes()
        var storage = InlinedFixedVector[Bytes](
            capacity=(tensor.num_elements() + shapes.__len__() + 2)
        )
        var magic_bytes = tobytes[DType.uint64](Self.MAGIC_NUMBER)
        storage[0] = magic_bytes

        for i in range(len(shapes)):
            var shape_bytes = tobytes[DType.int64](shapes[i])
            storage[i + 1] = shape_bytes

        var end_bytes = tobytes[DType.uint64](0x1A)
        storage[len(shapes) + 1] = end_bytes

        for i in range(tensor.num_elements()):
            var value = tensor.load(i)
            var bytes = tobytes[type](value)
            storage[i + 2 + len(shapes)] = bytes
        self.shapes = shapes
        self.storage = storage

    fn totensor[type: DType](inout self: Self) -> Tensor[type]:
        var shapes = shape(self.shapes)
        var tensor = Tensor[type](shapes)
        for i in range(tensor.num_elements()):
            var value = frombytes[type](self.storage[i + 2 + len(shapes)])
            tensor.store(i, value)
        return tensor

    fn write(self, path: String):
        with fopen(path, "wb") as file:
            for i in range(self.storage.capacity):
                file.write_bytes(self.storage[i])

    fn read(inout self, path: String):
        with fopen(path, "rb") as file:
            var magic_buffer = Bytes(file.read_bytes(NBytes))
            var magic_number = frombytes[DType.uint64](magic_buffer)
            if magic_number != Self.MAGIC_NUMBER:
                print("Invalid magic number")
                exit(1)

            var shapes = List[Int](capacity=26)
            while True:
                var shape_buffer = Bytes(file.read_bytes(NBytes))
                var shape_value = frombytes[DType.uint64](shape_buffer)
                if shape_value == 0x1A:
                    break
                shapes.append(int(shape_value))

            var num_elements = 1
            for i in range(shapes.size):
                num_elements *= shapes[i]

            var storage = InlinedFixedVector[Bytes](
                capacity=num_elements + shapes.__len__() + 2
            )
            storage[0] = tobytes[DType.uint64](Self.MAGIC_NUMBER)
            for i in range(len(shapes)):
                storage[i + 1] = tobytes[DType.uint64](shapes[i])
            storage[len(shapes) + 1] = tobytes[DType.uint64](0x1A)

            for i in range(num_elements):
                var element_buffer = Bytes(file.read_bytes(NBytes))
                storage[i + 2 + len(shapes)] = element_buffer
            self = Self(shapes, storage)

    @staticmethod
    fn read(path: String) -> Serialize:
        with fopen(path, "rb") as file:
            var magic_buffer = Bytes(file.read_bytes(NBytes))
            var magic_number = frombytes[DType.uint64](magic_buffer)
            if magic_number != Self.MAGIC_NUMBER:
                print("Invalid magic number")
                exit(1)

            var shapes = List[Int](capacity=26)
            while True:
                var shape_buffer = Bytes(file.read_bytes(NBytes))
                var shape_value = frombytes[DType.uint64](shape_buffer)
                if shape_value == 0x1A:
                    break
                shapes.append(int(shape_value))

            var num_elements = 1
            for i in range(shapes.size):
                num_elements *= shapes[i]

            var storage = InlinedFixedVector[Bytes](
                capacity=num_elements + shapes.__len__() + 2
            )
            storage[0] = tobytes[DType.uint64](Self.MAGIC_NUMBER)
            for i in range(len(shapes)):
                storage[i + 1] = tobytes[DType.uint64](shapes[i])
            storage[len(shapes) + 1] = tobytes[DType.uint64](0x1A)

            for i in range(num_elements):
                var element_buffer = Bytes(file.read(NBytes))
                storage[i + 2 + len(shapes)] = element_buffer
            return Self(shapes, storage)

    @staticmethod
    fn fromtensor[type: DType](tensor: Tensor[type]) -> Self:
        var shapes = tensor.shapes().Shapes()
        var storage = InlinedFixedVector[Bytes](
            capacity=(tensor.num_elements() + len(shapes) + 2)
        )
        var magic_bytes = tobytes[DType.uint64](Self.MAGIC_NUMBER)
        storage[0] = magic_bytes

        for i in range(len(shapes)):
            var shape_bytes = tobytes[DType.int64](shapes[i])
            storage[i + 1] = shape_bytes

        var end_bytes = tobytes[DType.uint64](0x1A)
        storage[len(shapes) + 1] = end_bytes

        for i in range(tensor.num_elements()):
            var value = tensor.load(i)
            var bytes = tobytes[type](value)
            storage[i + 2 + len(shapes)] = bytes
        return Self(shapes, storage)

    @staticmethod
    fn totensor[type: DType](bytes: Serialize) -> Tensor[type]:
        var shapes = shape(bytes.shapes)
        var tensor = Tensor[type](shapes)
        for i in range(tensor.num_elements()):
            var value = frombytes[type](bytes.storage[i + 2 + len(shapes)])
            tensor.store(i, value)
        return tensor

    fn list(self) -> List[Bytes]:
        var list_bytes = List[Bytes](capacity=self.storage.capacity)
        for i in range(self.storage.capacity):
            list_bytes[i] = self.storage[i]
        return list_bytes

# struct ckpt:
#     var filename : String
#     var path : String
#     var steps : Int
#     var type : String
#     var model : Module
