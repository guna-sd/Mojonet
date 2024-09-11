@register_passable("trivial")
struct StorageImpl:
    var ptr: DataPointer
    var size: UInt
    var type: DType
