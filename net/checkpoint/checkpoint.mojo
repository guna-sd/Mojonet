from net.nn.module import Module

struct Serialize:
    var storage : List[Bytes]

    fn __init__(inout self):
        self.storage = List[Bytes]()
    
    fn __init__(inout self, data : List[Bytes]):
        self.storage = data
    
    fn __init__(inout self, *data : Bytes):
        self = Serialize()
        for i in range(data.__len__()):
            self.storage.append(data[i])
    
    fn __init__(inout self, data : Bytes):
        self = Serialize()
        self.storage.append(data)

struct ckpt:
    var filename : String
    var path : String
    var steps : Int
    var type : String
    var model : Module