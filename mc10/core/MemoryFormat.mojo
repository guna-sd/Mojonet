@value
struct MemoryFormat:
    alias Contiguous = MemoryFormat(0)
    alias Preserve = MemoryFormat(1)
    alias ChannelsLast = MemoryFormat(2)
    alias ChannelsLast3d = MemoryFormat(3)
    alias NumOptions = MemoryFormat(4)

    var value: Int8

    @always_inline
    fn __init__(out self):
        self = MemoryFormat.Contiguous
    
    @no_inline
    fn __str__(self) -> String:
        """
        Gets the memory format in string.

        Returns:
            The memory format.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        return "MemoryFormat." + str(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        return hash(UInt8(self.value.cast[DType.uint8]()))
    
    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        if self == MemoryFormat.Contiguous:
            return writer.write("contiguous")
        if self == MemoryFormat.Preserve:
            return writer.write("preserve")
        if self == MemoryFormat.ChannelsLast:
            return writer.write("channels_last")
        if self == MemoryFormat.ChannelsLast3d:
            return writer.write("channels_last_3d")
        if self == MemoryFormat.NumOptions:
            return writer.write("num_options")
        return writer.write("unknown format")

    @staticmethod
    fn _from_str(device_str: String) -> MemoryFormat:
        if device_str.startswith("MemoryFormat."):
            return MemoryFormat._from_str(device_str.removeprefix("MemoryFormat."))
        
        elif device_str == "contiguous":
            return MemoryFormat.Contiguous
        elif device_str == "preserve":
            return MemoryFormat.Preserve
        elif device_str == "channels_last":
            return MemoryFormat.ChannelsLast
        elif device_str == "channels_last_3d":
            return MemoryFormat.ChannelsLast3d
        elif device_str == "num_options":
            return MemoryFormat.NumOptions
        else:
            return MemoryFormat.Contiguous

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        return self != rhs
    
    @always_inline("nodebug")
    fn is_contiguous(self) -> Bool:
        return self == Self.Contiguous

    @always_inline("nodebug")
    fn is_preserve(self) -> Bool:
        return self == Self.Preserve

    @always_inline("nodebug")
    fn is_channels_last(self) -> Bool:
        return self == Self.ChannelsLast

    @always_inline("nodebug")
    fn is_channels_last_3d(self) -> Bool:
        return self == Self.ChannelsLast3d

    @always_inline("nodebug")
    fn is_numopts(self) -> Bool:
        return self == Self.NumOptions

    @always_inline("nodebug")
    @staticmethod
    fn is_valid(value: Int8) -> Bool:
        return value >= 0 and value < 4