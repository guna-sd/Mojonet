@value
@register_passable("trivial")
struct Device(Stringable, Formattable, Representable, KeyElement):
    alias CPU = Device(0)
    alias CUDA = Device(1)
    alias MKLDNN = Device(2)
    alias OPENGL = Device(3)
    alias OPENCL = Device(4)
    alias IDEEP = Device(5)
    alias HIP = Device(6)
    alias FPGA = Device(7)
    alias MSNPU = Device(8)
    alias XLA = Device(9)
    alias Vulkan = Device(10)
    alias Metal = Device(11)
    alias XPU = Device(12)
    alias MLC = Device(13)
    alias Meta = Device(14)
    alias HPU = Device(15)
    alias COMPILE_TIME_MAX_DEVICE_TYPES = 16
    var value: Int8

    @always_inline
    fn __init__(inout self):
        self = Device.CPU
        
    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the Device.

        Returns:
            The name of the device.
        """
        return String.format_sequence(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the Device e.g. `"Device.cpu"`.

        Returns:
            The representation of the device.
        """
        return "Device." + str(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the Device.

        Returns:
            An integer hash value based on the Device's value.
        """
        return hash(UInt8(self.value.cast[DType.uint8]()))

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this device to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        if self == Device.CPU:
            return writer.write_str("cpu")
        if self == Device.CUDA:
            return writer.write_str("cuda")
        if self == Device.MKLDNN:
            return writer.write_str("MKLDNN")
        if self == Device.OPENGL:
            return writer.write_str("OPENGL")
        if self == Device.OPENCL:
            return writer.write_str("OPENCL")
        if self == Device.IDEEP:
            return writer.write_str("IDEAP")
        if self == Device.HIP:
            return writer.write_str("HIP")
        if self == Device.FPGA:
            return writer.write_str("FPGA")
        if self == Device.MSNPU:
            return writer.write_str("MSNPU")
        if self == Device.XLA:
            return writer.write_str("XLA")
        if self == Device.Vulkan:
            return writer.write_str("Vulkan")
        if self == Device.Metal:
            return writer.write_str("Metal")
        if self == Device.XPU:
            return writer.write_str("XPU")
        if self == Device.MLC:
            return writer.write_str("MLC")
        if self == Device.Meta:
            return writer.write_str("Meta")
        if self == Device.HPU:
            return writer.write_str("HPU")
        return writer.write_str("Unknown device")

    @staticmethod
    fn _from_str(device_str: String) -> Device:
        """Construct a Device from a string.
        
        Args:
            device_str: The name of the Device.
        """
        if device_str.startswith("Device."):
            return Self._from_str(device_str.removeprefix("Device."))
        elif device_str == "cpu":
            return Device.CPU
        elif device_str == "cuda":
            return Device.CUDA
        elif device_str == "xpu":
            return Device.XPU
        elif device_str == "mkldnn":
            return Device.MKLDNN
        elif device_str == "opengl":
            return Device.OPENGL
        elif device_str == "opencl":
            return Device.OPENCL
        elif device_str == "ideep":
            return Device.IDEEP
        elif device_str == "hip":
            return Device.HIP
        elif device_str == "fpga":
            return Device.FPGA
        elif device_str == "msnpu":
            return Device.MSNPU
        elif device_str == "xla":
            return Device.XLA
        elif device_str == "vulkan":
            return Device.Vulkan
        elif device_str == "mlc":
            return Device.MLC
        elif device_str == "meta":
            return Device.Meta
        elif device_str == "hpu":
            return Device.HPU
        else:
            return Device.CPU

    @always_inline("nodebug")
    fn __eq__(self, rhs: Device) -> Bool:
        """Compares one Device to another for equality.

        Args:
            rhs: The Device to compare against.

        Returns:
            True if the Devices are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Device) -> Bool:
        """Compares one Device to another for inequality.

        Args:
            rhs: The Device to compare against.

        Returns:
            False if the Devices are the same and True otherwise.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Device) -> Bool:
        """Compares one Device to another for equality.

        Args:
            rhs: The Device to compare against.

        Returns:
            True if the Devices are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Device) -> Bool:
        """Compares one Device to another for inequality.

        Args:
            rhs: The Device to compare against.

        Returns:
            True if the Devices are the same and False otherwise.
        """
        return self != rhs

    @always_inline("nodebug")
    fn is_cpu(self) -> Bool:
        """Checks if the Device is a CPU.

        Returns:
            True if the Device is a CPU, False otherwise.
        """
        return self == Device.CPU

    @always_inline("nodebug")
    fn is_cuda(self) -> Bool:
        """Checks if the Device is a CUDA device.

        Returns:
            True if the Device is a CUDA device, False otherwise.
        """
        return self == Device.CUDA

    @always_inline("nodebug")
    fn is_mkldnn(self) -> Bool:
        """Checks if the Device is an MKLDNN device.

        Returns:
            True if the Device is an MKLDNN device, False otherwise.
        """
        return self == Device.MKLDNN

    @always_inline("nodebug")
    fn is_opengl(self) -> Bool:
        """Checks if the Device is an OpenGL device.

        Returns:
            True if the Device is an OpenGL device, False otherwise.
        """
        return self == Device.OPENGL

    @always_inline("nodebug")
    fn is_opencl(self) -> Bool:
        """Checks if the Device is an OpenCL device.

        Returns:
            True if the Device is an OpenCL device, False otherwise.
        """
        return self == Device.OPENCL

    @always_inline("nodebug")
    fn is_ideep(self) -> Bool:
        """Checks if the Device is an IDEEP device.

        Returns:
            True if the Device is an IDEEP device, False otherwise.
        """
        return self == Device.IDEEP

    @always_inline("nodebug")
    fn is_hip(self) -> Bool:
        """Checks if the Device is a HIP device.

        Returns:
            True if the Device is a HIP device, False otherwise.
        """
        return self == Device.HIP

    @always_inline("nodebug")
    fn is_fpga(self) -> Bool:
        """Checks if the Device is an FPGA device.

        Returns:
            True if the Device is an FPGA device, False otherwise.
        """
        return self == Device.FPGA

    @always_inline("nodebug")
    fn is_msnpu(self) -> Bool:
        """Checks if the Device is an MSNPU device.

        Returns:
            True if the Device is an MSNPU device, False otherwise.
        """
        return self == Device.MSNPU

    @always_inline("nodebug")
    fn is_xla(self) -> Bool:
        """Checks if the Device is an XLA device.

        Returns:
            True if the Device is an XLA device, False otherwise.
        """
        return self == Device.XLA

    @always_inline("nodebug")
    fn is_vulkan(self) -> Bool:
        """Checks if the Device is a Vulkan device.

        Returns:
            True if the Device is a Vulkan device, False otherwise.
        """
        return self == Device.Vulkan

    @always_inline("nodebug")
    fn is_metal(self) -> Bool:
        """Checks if the Device is a Metal device.

        Returns:
            True if the Device is a Metal device, False otherwise.
        """
        return self == Device.Metal

    @always_inline("nodebug")
    fn is_xpu(self) -> Bool:
        """Checks if the Device is an XPU device.

        Returns:
            True if the Device is an XPU device, False otherwise.
        """
        return self == Device.XPU

    @always_inline("nodebug")
    fn is_mlc(self) -> Bool:
        """Checks if the Device is an MLC device.

        Returns:
            True if the Device is an MLC device, False otherwise.
        """
        return self == Device.MLC

    @always_inline("nodebug")
    fn is_meta(self) -> Bool:
        """Checks if the Device is a Meta device.

        Returns:
            True if the Device is a Meta device, False otherwise.
        """
        return self == Device.Meta

    @always_inline("nodebug")
    fn is_hpu(self) -> Bool:
        """Checks if the Device is an HPU device.

        Returns:
            True if the Device is an HPU device, False otherwise.
        """
        return self == Device.HPU
