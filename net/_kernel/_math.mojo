from net import Tensor

alias e     : FloatLiteral = 2.7182818284_5904523536_0287471352_6624977572_4709369995_9574966967_6277240766_3035354759_4571382178_5251664274 # exp(1)
alias pi    : FloatLiteral = 3.1415926535_8979323846_2643383279_5028841971_6939937510_5820974944_5923078164_0628620899_8628034825_3421170679 # acos(-1)
alias tau   : FloatLiteral = 6.2831853071_7958647692_5286766559_0057683943_3879875021_1641949889_1846156328_1257241799_7256069650_6842341359 # pi*2
alias hfpi  : FloatLiteral = 1.5707963267_9489661923_1321691639_7514420985_8469968755_2910487472_2961539082_0314310449_9314017412_6710585339 # pi/2
alias trpi  : FloatLiteral = 1.0471975511_9659774615_4214461093_1676280657_2313312503_5273658314_8641026054_6876206966_6209344941_7807056893 # pi/3
alias qtpi  : FloatLiteral = 0.7853981633_9744830961_5660845819_8757210492_9234984377_6455243736_1480769541_0157155224_9657008706_3355292669 # pi/4
alias phi   : FloatLiteral = 1.6180339887_4989484820_4586834365_6381177203_0917980576_2862135448_6227052604_6281890244_9707207204_1893911374 # sqrt(5)+1 / 2
alias pho   : FloatLiteral = 0.6180339887_4989484820_4586834365_6381177203_0917980576_2862135448_6227052604_6281890244_9707207204_1893911374 # sqrt(5)-1 / 2
alias rt2   : FloatLiteral = 1.4142135623_7309504880_1688724209_6980785696_7187537694_8073176679_7379907324_7846210703_8850387534_3276415727 # sqrt(2)
alias trh   : FloatLiteral = 0.8660254037_8443864676_3723170752_9361834714_0262690519_0314027903_4897259665_0845440001_8540573093_3786242878 # sqrt(3)/2
alias twrt2 : FloatLiteral = 1.0594630943_5929526456_1825294946_3417007792_0431749418_5628559208_4314587616_4606325572_2383768376_8639455690 # pow(2, 1/12)
alias ln2   : FloatLiteral = 0.6931471805_5994530941_7232121458_1765680755_0013436025_5254120680_0094933936_2196969471_5605863326_9964186875 # ln(2)
alias omega : FloatLiteral = 0.5671432904_0978387299_9968662210_3555497538_1578718651_2508135131_0792230457_9308668456_6693219446_9617522945 # lw(1)
alias gamma : FloatLiteral = 0.5772156649_0153286060_6512090082_4024310421_5933593992_3598805767_2348848677_2677766467_0936947063_2917467495 # -diff[lgamma](1)

fn abs[type : DType](value : SIMD[type,1]) -> SIMD[type,1]:
    """
    Find the absolute value of a number.
    """
    return -value if value < 0.0 else value

fn max[type : DType](value : Tensor[type]) -> SIMD[type,1]:
    """
    Find the maximum value in the Tensor.
    """
    if value.num_elements() == 0:
        print("list is empty")
        return 0
    if value.num_elements() == 1:
        return value[0]
    var j = value[0]
    for i in range(value.num_elements()):
        if abs[type](value[i]) > abs[type](j):
            j = value[i]
    return j

fn min[type : DType](value : Tensor[type]) -> SIMD[type,1]:
    """
    Find the minimum value in the list.
    """
    if value.num_elements() == 0:
        print("list is empty")
        return 0
    if value.num_elements() == 1:
        return value[0]

    var j = value[0]
    for i in range(value.num_elements()):
        if abs(value[i]) < abs(j):
            j = value[i]
    return j

fn add[type : DType](owned first: SIMD[type,1], owned second: SIMD[type,1]) -> SIMD[type,1]:
    """
    Implementation of addition of integer.
    """

    while second != 0:
        var c = first & second
        first ^= second
        second = c << 1
    return first

fn add[type : DType](owned first: Tensor[type], owned second: Tensor[type]) -> Tensor[type]:
    var out : Tensor[type] = Tensor[type](first.shape)
    if first.shape._rank == second.shape._rank:
        for i in range(first.shape.num_elements):
            out[i] = add[type](first[i], second[i])
        return out
    else:
        print("adding two different tensors is not supported")
        return out

fn pow[type : DType, nelts : Int](base: SIMD[type,nelts], exponent: Int) -> SIMD[type,nelts]:
    """
    Computes a^b recursively, where a is the base and b is the exponent.
    """
    if exponent < 0:
        print("Exponent must be a non-negative integer")

    if exponent == 0:
        return 1

    if exponent % 2 == 1:
        return pow(base, exponent - 1) * base

    var b = pow(base, exponent // 2)
    return b * b

fn multiply[type : DType](owned a: SIMD[type,1], owned b: SIMD[type,1]) -> SIMD[type,1]:
    """
    Multiply 'a' and 'b' using bitwise multiplication.
    """
    var res : SIMD[type,1] = 0
    while b > 0:
        if b & 1:
            res += a

        a += a
        b >>= 1

    return res