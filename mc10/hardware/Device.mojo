@fieldwise_init
@register_passable("trivial")
struct DeviceType(AnyType, Copyable, EqualityComparable, Hashable, KeyElement, Representable, Stringable, Writable):
    alias CPU = DeviceType(0)
    alias CUDA = DeviceType(1)
    alias MKLDNN = DeviceType(2)
    alias OPENGL = DeviceType(3)
    alias OPENCL = DeviceType(4)
    alias IDEEP = DeviceType(5)
    alias HIP = DeviceType(6)
    alias AMD = DeviceType(6)
    alias FPGA = DeviceType(7)
    alias MSNPU = DeviceType(8)
    alias XLA = DeviceType(9)
    alias Vulkan = DeviceType(10)
    alias Metal = DeviceType(11)
    alias XPU = DeviceType(12)
    alias MLC = DeviceType(13)
    alias Meta = DeviceType(14)
    alias HPU = DeviceType(15)
    alias COMPILE_TIME_MAX_DEVICEType_TYPES = 16
    var value: Int8

    @always_inline
    fn __init__(out self):
        self = DeviceType.CPU

    @no_inline
    fn __str__(self) -> String:
        """Gets the name of the DeviceType.

        Returns:
            The name of the deviceType.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Gets the representation of the DeviceType e.g. `"DeviceType.cpu"`.

        Returns:
            The representation of the deviceType.
        """
        return "DeviceType." + String(self)

    @always_inline("nodebug")
    fn __hash__(self) -> UInt:
        """Computes the hash value for the DeviceType.

        Returns:
            An integer hash value based on the DeviceType's value.
        """
        return hash(UInt8(self.value.cast[DType.uint8]()))

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this deviceType to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        if self == DeviceType.CPU:
            return writer.write("cpu")
        if self == DeviceType.CUDA:
            return writer.write("cuda")
        if self == DeviceType.MKLDNN:
            return writer.write("MKLDNN")
        if self == DeviceType.OPENGL:
            return writer.write("OPENGL")
        if self == DeviceType.OPENCL:
            return writer.write("OPENCL")
        if self == DeviceType.IDEEP:
            return writer.write("IDEEP")
        if self == DeviceType.AMD:
            return writer.write("AMD")
        if self == DeviceType.FPGA:
            return writer.write("FPGA")
        if self == DeviceType.MSNPU:
            return writer.write("MSNPU")
        if self == DeviceType.XLA:
            return writer.write("XLA")
        if self == DeviceType.Vulkan:
            return writer.write("Vulkan")
        if self == DeviceType.Metal:
            return writer.write("Metal")
        if self == DeviceType.XPU:
            return writer.write("XPU")
        if self == DeviceType.MLC:
            return writer.write("MLC")
        if self == DeviceType.Meta:
            return writer.write("Meta")
        if self == DeviceType.HPU:
            return writer.write("HPU")
        return writer.write("Unknown deviceType")

    @staticmethod
    fn _from_str(deviceType_str: String) -> DeviceType:
        """Construct a DeviceType from a string.

        Args:
            deviceType_str: The name of the DeviceType.
        """
        if deviceType_str.startswith("DeviceType."):
            return Self._from_str(deviceType_str.removeprefix("DeviceType."))
        elif deviceType_str == "cpu":
            return DeviceType.CPU
        elif deviceType_str == "cuda":
            return DeviceType.CUDA
        elif deviceType_str == "xpu":
            return DeviceType.XPU
        elif deviceType_str == "mkldnn":
            return DeviceType.MKLDNN
        elif deviceType_str == "opengl":
            return DeviceType.OPENGL
        elif deviceType_str == "opencl":
            return DeviceType.OPENCL
        elif deviceType_str == "ideep":
            return DeviceType.IDEEP
        elif deviceType_str == "hip" or deviceType_str == "amd":
            return DeviceType.AMD
        elif deviceType_str == "fpga":
            return DeviceType.FPGA
        elif deviceType_str == "msnpu":
            return DeviceType.MSNPU
        elif deviceType_str == "xla":
            return DeviceType.XLA
        elif deviceType_str == "vulkan":
            return DeviceType.Vulkan
        elif deviceType_str == "mlc":
            return DeviceType.MLC
        elif deviceType_str == "meta":
            return DeviceType.Meta
        elif deviceType_str == "hpu":
            return DeviceType.HPU
        else:
            return DeviceType.CPU

    @always_inline("nodebug")
    fn __eq__(self, rhs: DeviceType) -> Bool:
        """Compares one DeviceType to another for equality.

        Args:
            rhs: The DeviceType to compare against.

        Returns:
            True if the DeviceTypes are the same and False otherwise.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: DeviceType) -> Bool:
        """Compares one DeviceType to another for inequality.

        Args:
            rhs: The DeviceType to compare against.

        Returns:
            False if the DeviceTypes are the same and True otherwise.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: DeviceType) -> Bool:
        """Compares one DeviceType to another for equality.

        Args:
            rhs: The DeviceType to compare against.

        Returns:
            True if the DeviceTypes are the same and False otherwise.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: DeviceType) -> Bool:
        """Compares one DeviceType to another for inequality.

        Args:
            rhs: The DeviceType to compare against.

        Returns:
            True if the DeviceTypes are the same and False otherwise.
        """
        return self != rhs

    @always_inline("nodebug")
    fn is_cpu(self) -> Bool:
        """Checks if the DeviceType is a CPU.

        Returns:
            True if the DeviceType is a CPU, False otherwise.
        """
        return self == DeviceType.CPU

    @always_inline("nodebug")
    fn is_cuda(self) -> Bool:
        """Checks if the DeviceType is a CUDA deviceType.

        Returns:
            True if the DeviceType is a CUDA deviceType, False otherwise.
        """
        return self == DeviceType.CUDA

    @always_inline("nodebug")
    fn is_mkldnn(self) -> Bool:
        """Checks if the DeviceType is an MKLDNN deviceType.

        Returns:
            True if the DeviceType is an MKLDNN deviceType, False otherwise.
        """
        return self == DeviceType.MKLDNN

    @always_inline("nodebug")
    fn is_opengl(self) -> Bool:
        """Checks if the DeviceType is an OpenGL deviceType.

        Returns:
            True if the DeviceType is an OpenGL deviceType, False otherwise.
        """
        return self == DeviceType.OPENGL

    @always_inline("nodebug")
    fn is_opencl(self) -> Bool:
        """Checks if the DeviceType is an OpenCL deviceType.

        Returns:
            True if the DeviceType is an OpenCL deviceType, False otherwise.
        """
        return self == DeviceType.OPENCL

    @always_inline("nodebug")
    fn is_ideep(self) -> Bool:
        """Checks if the DeviceType is an IDEEP deviceType.

        Returns:
            True if the DeviceType is an IDEEP deviceType, False otherwise.
        """
        return self == DeviceType.IDEEP

    @always_inline("nodebug")
    fn is_hip(self) -> Bool:
        """Checks if the DeviceType is a HIP deviceType.

        Returns:
            True if the DeviceType is a HIP deviceType, False otherwise.
        """
        return self == DeviceType.HIP

    @always_inline("nodebug")
    fn is_amd(self) -> Bool:
        """Checks if the DeviceType is a AMD deviceType.

        Returns:
            True if the DeviceType is a AMD deviceType, False otherwise.
        """
        return self == DeviceType.AMD

    @always_inline("nodebug")
    fn is_fpga(self) -> Bool:
        """Checks if the DeviceType is an FPGA deviceType.

        Returns:
            True if the DeviceType is an FPGA deviceType, False otherwise.
        """
        return self == DeviceType.FPGA

    @always_inline("nodebug")
    fn is_msnpu(self) -> Bool:
        """Checks if the DeviceType is an MSNPU deviceType.

        Returns:
            True if the DeviceType is an MSNPU deviceType, False otherwise.
        """
        return self == DeviceType.MSNPU

    @always_inline("nodebug")
    fn is_xla(self) -> Bool:
        """Checks if the DeviceType is an XLA deviceType.

        Returns:
            True if the DeviceType is an XLA deviceType, False otherwise.
        """
        return self == DeviceType.XLA

    @always_inline("nodebug")
    fn is_vulkan(self) -> Bool:
        """Checks if the DeviceType is a Vulkan deviceType.

        Returns:
            True if the DeviceType is a Vulkan deviceType, False otherwise.
        """
        return self == DeviceType.Vulkan

    @always_inline("nodebug")
    fn is_metal(self) -> Bool:
        """Checks if the DeviceType is a Metal deviceType.

        Returns:
            True if the DeviceType is a Metal deviceType, False otherwise.
        """
        return self == DeviceType.Metal

    @always_inline("nodebug")
    fn is_xpu(self) -> Bool:
        """Checks if the DeviceType is an XPU deviceType.

        Returns:
            True if the DeviceType is an XPU deviceType, False otherwise.
        """
        return self == DeviceType.XPU

    @always_inline("nodebug")
    fn is_mlc(self) -> Bool:
        """Checks if the DeviceType is an MLC deviceType.

        Returns:
            True if the DeviceType is an MLC deviceType, False otherwise.
        """
        return self == DeviceType.MLC

    @always_inline("nodebug")
    fn is_meta(self) -> Bool:
        """Checks if the DeviceType is a Meta deviceType.

        Returns:
            True if the DeviceType is a Meta deviceType, False otherwise.
        """
        return self == DeviceType.Meta

    @always_inline("nodebug")
    fn is_hpu(self) -> Bool:
        """Checks if the DeviceType is an HPU deviceType.

        Returns:
            True if the DeviceType is an HPU deviceType, False otherwise.
        """
        return self == DeviceType.HPU

    @parameter
    @always_inline("nodebug")
    fn address_space(self) -> AddressSpace:
        """
        Returns the address space corresponding to the DeviceType.

        For example:
        - CPU deviceTypes use AddressSpace(0)
        - CUDA/HIP/AMD deviceTypes use AddressSpace(1)
        - Other deviceTypes default to AddressSpace(0)

        Returns:
            An AddressSpace instance initialized with an integer tag.
        """
        if self == DeviceType.CPU:
            return AddressSpace(0)
        elif (
            self == DeviceType.CUDA
            or self == DeviceType.HIP
            or self == DeviceType.AMD
        ):
            return AddressSpace(1)

        else:
            return AddressSpace(0)


struct Target:
    alias __mlir_type = __mlir_type.`!kgen.target`

    alias A100 = Target(_get_a100_target())
    alias A10 = Target(_get_a10_target())
    alias OrinNano = Target(_get_orin_nano_target())
    alias H100 = Target(_get_h100_target())
    alias L4 = Target(_get_l4_target())
    alias B100 = Target(_get_b100_target())

    var _target: Self.__mlir_type

    fn __init__(out self, target: Self.__mlir_type):
        self._target = target

    @always_inline("nodebug")
    fn target(self) -> Self.__mlir_type:
        """
        Returns the current target configuration.

        Returns:
            The current MLIR target configuration.
        """
        return self._target

    @staticmethod
    @always_inline("nodebug")
    fn _get_empty_target() -> Self.__mlir_type:
        """
        Creates an empty target configuration for when no GPU is available.

        Returns:
            An empty MLIR target configuration.
        """
        return __mlir_attr[
            `#kgen.target<triple = "", `,
            `arch = "", `,
            `features = "", `,
            `data_layout="",`,
            `simd_bit_width = 0,`,
            `index_bit_width = 0`,
            `> : !kgen.target`,
        ]


fn _get_a100_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA A100 GPU.

    Returns:
        MLIR target configuration for A100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_80", `,
        `features = "+ptx81,+sm_80", `,
        `tune_cpu = "sm_80", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


fn _get_a10_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA A10 GPU.

    Returns:
        MLIR target configuration for A10.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


fn _get_orin_nano_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA Jetson Orin Nano GPU.

    Returns:
        MLIR target configuration for Orin Nano.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_87", `,
        `features = "+ptx81,+sm_87", `,
        `tune_cpu = "sm_87", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


fn _get_h100_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA H100 GPU.

    Returns:
        MLIR target configuration for H100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_l4_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA L4 GPU.

    Returns:
        MLIR target configuration for L4.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_89", `,
        `features = "+ptx81,+sm_89", `,
        `tune_cpu = "sm_89", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128,`,
        `index_bit_width = 64`,
        `> : !kgen.target`,
    ]


fn _get_b100_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA B100 GPU.

    Returns:
        MLIR target configuration for B100.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_100a", `,
        `features = "+ptx86,+sm_100a", `,
        `tune_cpu = "sm_100a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_mi300x_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for AMD MI300X GPU.

    Returns:
        MLIR target configuration for MI300X.
    """

    return __mlir_attr[
        `#kgen.target<triple = "amdgcn-amd-amdhsa", `,
        `arch = "gfx942", `,
        `features = "", `,
        `data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_rtx2060_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA RTX 2060 GPU.

    Returns:
        MLIR target configuration for RTX 2060.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_75", `,
        `features = "+ptx63,+sm_75", `,
        `tune_cpu = "sm_75", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


fn _get_rtx5090_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA RTX5090 GPU.

    Returns:
        MLIR target configuration for RTX5090.
    """

    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_120a", `,
        `features = "+ptx86,+sm_120a", `,
        `tune_cpu = "sm_120a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]



## TODO: Test this target
fn _get_rtx2050_target() -> __mlir_type.`!kgen.target`:
    """
    Creates an MLIR target configuration for NVIDIA RTX 2050 GPU.

    Returns:
        MLIR target configuration for RTX 2050.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_86", `,
        `features = "+ptx81,+sm_86", `,
        `tune_cpu = "sm_86", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]


