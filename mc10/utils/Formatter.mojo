# TODO: still under construction... Not really important for now...


@fieldwise_init
struct __PrinterOptions:
    var precision: Int
    var threshold: Float32
    var edgeitems: Int
    var linewidth: Int
    var max_width: Int
    var sci_mode: Bool

    fn __init__(out self, sci_mode: Bool):
        self = __PrinterOptions(
            precision=4,
            threshold=1000,
            edgeitems=3,
            linewidth=80,
            max_width=1,
            sci_mode=sci_mode,
        )


struct TensorFormatter(Writable):
    alias TensorStart = "Tensor("
    alias TensorEnd = ")"
    alias SquareBracketL = "["
    alias SquareBracketR = "]"
    alias Truncation = " ...,"
    alias CompactMaxElemsToPrint = 19
    alias CompactElemPerSide = 4

    var Format: __PrinterOptions

    fn write_scalar[W: Writer](self, mut buffer: W):
        ...

    fn write_int[W: Writer](self, mut buffer: W):
        ...

    fn write_float[W: Writer](self, mut buffer: W):
        ...

    fn write_double[W: Writer](self, mut buffer: W):
        ...

    fn write_1d[W: Writer](self, mut buffer: W):
        ...

    fn write_to[W: Writer](self, mut buffer: W):
        ...
