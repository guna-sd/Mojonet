from net.tensor.utils import shape, handle_issue


@always_inline("nodebug")
fn check_shape(a: shape, b: shape) -> Bool:
    """
    Checks whether two shapes are compatible for matrix multiplication.

    Args:
        a: The shape of the first tensor.
        b: The shape of the second tensor.

    Returns:
        A Boolean value indicating whether the shapes are compatible for matrix multiplication.
    """
    if a.rank < 1 or b.rank < 1:
        return False

    if a.rank == 1 and b.rank == 1:
        return a[0] == b[0]

    if a.rank == 1 and b.rank > 1:
        return a[0] == b[b.rank - 2]

    if b.rank == 1 and a.rank > 1:
        return a[a.rank - 1] == b[0]

    if a[-1] != b[-2]:
        return False

    return True


@always_inline("nodebug")
fn calculate_shapes(shape1: shape, shape2: shape) -> shape:
    """
    Calculates the resulting shape of the matrix multiplication operation between two input shapes.

    Args:
        shape1: The shape of the first tensor.
        shape2: The shape of the second tensor.

    Returns:
        The resulting shape of the matrix multiplication operation.
    """
    if not check_shape(shape1, shape2):
        handle_issue("Tensors cannot be multiplied due to incompatible shapes.")

    var batch_dims = List[Int]()
    var max_batch_rank = max(shape1.rank - 2, shape2.rank - 2)
    for i in range(max_batch_rank):
        var dim1 = shape1[i] if i < shape1.rank - 2 else 1
        var dim2 = shape2[i] if i < shape2.rank - 2 else 1
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            handle_issue(
                "Incompatible dimensions at index"
                + str(i)
                + ":"
                + str(dim1)
                + "vs"
                + str(dim2)
            )

        batch_dims.append(max(dim1, dim2))

    if shape1.rank > 1 and shape2.rank > 1:
        batch_dims.append(shape1[shape1.rank - 2])
        batch_dims.append(shape2[shape2.rank - 1])
    elif shape1.rank > 1 and shape2.rank == 1:
        batch_dims.append(shape1[shape1.rank - 2])
    elif shape1.rank == 1 and shape2.rank > 1:
        batch_dims.append(shape2[shape2.rank - 1])

    return shape(batch_dims)
