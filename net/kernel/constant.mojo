struct Constants:
    """Commonly used mathematical constants."""

    # Euler's number
    alias e = 2.718281828459045235360287471352662
    """`e` is Euler's number, the base of natural logarithms, approximately 2.71828."""

    # Logarithms of e
    alias log2e = 1.442695040888963407359924681001892137426646
    """`log2e` is the base-2 logarithm of `e`, approximately 1.44270."""
    alias log10e = 0.434294481903251827651128918916605
    """`log10e` is the base-10 logarithm of `e`, approximately 0.43429."""

    # Pi and related constants
    alias pi = 3.141592653589793238462643383279502
    """`pi` is the ratio of the circumference of a circle to its diameter, approximately 3.14159."""
    alias tau = 6.283185307179586476925286766559
    """`tau` is the ratio of the circumference of a circle to its radius, equal to 2π, approximately 6.28319."""
    alias half_pi = 1.5707963267948966192313216916398
    """`half_pi` is half of `pi`, approximately 1.57080."""
    alias sqrt_half_pi = 1.2533141373155002512078826424055
    """`sqrt_half_pi` is the square root of half of `pi`, approximately 1.25331."""

    # Golden ratio and its reciprocal
    alias phi = 1.618033988749894848204586834365638
    """`phi` is the golden ratio, approximately 1.61803."""
    alias golden_ratio = 1.618033988749894848204586834365638
    """`golden_ratio` approximately 1.61803."""
    alias reciprocal_phi = 0.618033988749894848204586834365638
    """`reciprocal_phi` is the reciprocal of the golden ratio, approximately 0.61803."""

    # Square roots
    alias sqrt2 = 1.414213562373095048801688724209698
    """`sqrt2` is the square root of 2, approximately 1.41421."""
    alias sqrt3 = 1.732050807568877293527446341505872
    """`sqrt3` is the square root of 3, approximately 1.73205."""
    alias sqrt5 = 2.236067977499789696409173668731276
    """`sqrt5` is the square root of 5, approximately 2.23607."""

    # Natural logarithms
    alias ln2 = 0.693147180559945309417232121458176
    """`ln2` is the natural logarithm of 2, approximately 0.69315."""
    alias ln10 = 2.302585092994045684017991454684364
    """`ln10` is the natural logarithm of 10, approximately 2.30259."""

    # Other constants
    alias omega = 0.567143290409783872999968662210355
    """`omega` is the Lambert W function at 1, approximately 0.56714."""
    alias gamma = 0.577215664901532860606512090082402
    """`gamma` is the Euler-Mascheroni constant, approximately 0.57722."""
    alias euler = 0.577215664901532860606512090082402
    """`euler` is the Euler-Mascheroni constant, approximately 0.57722."""
    alias catalan = 0.915965594177219015054603514932384
    """`catalan` is Catalan's constant, approximately 0.91597."""
    alias apery = 1.202056903159594285399738161511449
    """`apery` is Apéry's constant, approximately 1.20206."""


@parameter
@always_inline("nodebug")
fn gamma[dtype: DType]() -> Scalar[dtype]:
    """Euler-Mascheroni constant as the specified data type."""
    return Scalar[dtype](Constants.gamma)


@parameter
@always_inline("nodebug")
fn euler[dtype: DType]() -> Scalar[dtype]:
    """Euler-Mascheroni constant as the specified data type."""
    return Scalar[dtype](Constants.gamma)


@parameter
@always_inline("nodebug")
fn e[dtype: DType]() -> Scalar[dtype]:
    """Returns Euler's number as the specified data type."""
    return Scalar[dtype](Constants.e)


@parameter
@always_inline("nodebug")
fn pi[dtype: DType]() -> Scalar[dtype]:
    """Returns Pi as the specified data type."""
    return Scalar[dtype](Constants.pi)


@parameter
@always_inline("nodebug")
fn sqrt2[dtype: DType]() -> Scalar[dtype]:
    """Returns the square root of 2 as the specified data type."""
    return Scalar[dtype](Constants.sqrt2)


@parameter
@always_inline("nodebug")
fn ln2[dtype: DType]() -> Scalar[dtype]:
    """Returns the natural logarithm of 2 as the specified data type."""
    return Scalar[dtype](Constants.ln2)


@parameter
@always_inline("nodebug")
fn log2e[dtype: DType]() -> Scalar[dtype]:
    """Returns the base-2 logarithm of Euler's number as the specified data type.
    """
    return Scalar[dtype](Constants.log2e)


@parameter
@always_inline("nodebug")
fn log10e[dtype: DType]() -> Scalar[dtype]:
    """Returns the base-10 logarithm of Euler's number as the specified data type.
    """
    return Scalar[dtype](Constants.log10e)


@parameter
@always_inline("nodebug")
fn tau[dtype: DType]() -> Scalar[dtype]:
    """Returns Tau (2π) as the specified data type."""
    return Scalar[dtype](Constants.tau)


@parameter
@always_inline("nodebug")
fn half_pi[dtype: DType]() -> Scalar[dtype]:
    """Returns half of Pi as the specified data type."""
    return Scalar[dtype](Constants.half_pi)


@parameter
@always_inline("nodebug")
fn sqrt_half_pi[dtype: DType]() -> Scalar[dtype]:
    """Returns the square root of half of Pi as the specified data type."""
    return Scalar[dtype](Constants.sqrt_half_pi)


@parameter
@always_inline("nodebug")
fn phi[dtype: DType]() -> Scalar[dtype]:
    """Returns the golden ratio as the specified data type."""
    return Scalar[dtype](Constants.phi)


@parameter
@always_inline("nodebug")
fn golden_ratio[dtype: DType]() -> Scalar[dtype]:
    """Returns the golden ratio as the specified data type."""
    return Scalar[dtype](Constants.phi)


@parameter
@always_inline("nodebug")
fn reciprocal_phi[dtype: DType]() -> Scalar[dtype]:
    """Returns the reciprocal of the golden ratio as the specified data type."""
    return Scalar[dtype](Constants.reciprocal_phi)


@parameter
@always_inline("nodebug")
fn sqrt3[dtype: DType]() -> Scalar[dtype]:
    """Returns the square root of 3 as the specified data type."""
    return Scalar[dtype](Constants.sqrt3)


@parameter
@always_inline("nodebug")
fn sqrt5[dtype: DType]() -> Scalar[dtype]:
    """Returns the square root of 5 as the specified data type."""
    return Scalar[dtype](Constants.sqrt5)


@parameter
@always_inline("nodebug")
fn ln10[dtype: DType]() -> Scalar[dtype]:
    """Returns the natural logarithm of 10 as the specified data type."""
    return Scalar[dtype](Constants.ln10)


@parameter
@always_inline("nodebug")
fn omega[dtype: DType]() -> Scalar[dtype]:
    """Returns the Lambert W function at 1 as the specified data type."""
    return Scalar[dtype](Constants.omega)


@parameter
@always_inline("nodebug")
fn catalan[dtype: DType]() -> Scalar[dtype]:
    """Returns Catalan's constant as the specified data type."""
    return Scalar[dtype](Constants.catalan)


@parameter
@always_inline("nodebug")
fn apery[dtype: DType]() -> Scalar[dtype]:
    """Returns Apéry's constant as the specified data type."""
    return Scalar[dtype](Constants.apery)
