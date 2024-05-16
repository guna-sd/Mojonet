struct Emmbedding[T : DType = DType.float32]:
    var num_vocab : Int
    var num_embeddings : Int
    var weights : Tensor[T]

    fn __init__(inout self, num_vocab : Int, num_embeddings : Int):
        self.num_vocab = num_vocab
        self.num_embeddings = num_embeddings

        self.weights = Tensor[T](self.num_vocab, self.num_embeddings)
        self.weights.rand()
    
    fn __copyinit__(inout self, other : Self):
        self.num_vocab = other.num_vocab
        self.num_embeddings = other.num_embeddings
        self.weights = other.weights
    
    fn __moveinit__(inout self, owned other : Self):
        self.num_vocab = other.num_vocab
        self.num_embeddings = other.num_embeddings
        self.weights = other.weights