from net.tensor.utils import get_broadcast_index, broadcast_shapes, is_compatible
from net.utils import handle_issue
from net.tensor import TensorType
from algorithm import vectorize

struct Operation[dtype: DType]:
    alias nelts = simdwidthof[dtype]()

    @always_inline("nodebug")
    @staticmethod
    fn tensor_operation[func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],](tensor1: TensorType[dtype], tensor2: TensorType[dtype]) -> TensorType[dtype]:
        """Performs an element-wise operation on two tensors of same shapes."""
        constrained[dtype.is_numeric(), "dtype must be numeric"]()
        
        var result_shape = broadcast_shapes(tensor1.shape, tensor2.shape)
        var result = TensorType[dtype](result_shape)
        var num_elements = result.shape.num_elements

        if tensor1.shape == tensor2.shape:
            @parameter
            fn operation[nelts: Int](idx: Int):
                result.store[nelts](
                    idx, func[dtype, nelts](tensor1.load[nelts](idx), tensor2.load[nelts](idx))
                )

            vectorize[operation, Self.nelts](num_elements - (num_elements % Self.nelts))

            for i in range(num_elements - (num_elements % Self.nelts), num_elements):
                result.store[nelts=1](i, func(tensor1.load[nelts=1](i), tensor2.load[nelts=1](i)))
            return result

        @parameter
        fn vec_op[nelts: Int](i: Int):
            var flat_index1 = get_broadcast_index(i, tensor1.shape, result_shape)
            var flat_index2 = get_broadcast_index(i, tensor2.shape, result_shape)

            result.store[nelts](
                i,
                func[dtype, nelts](
                    tensor1.load[nelts](flat_index1),
                    tensor2.load[nelts](flat_index2),
                ),
            )

        vectorize[vec_op, Self.nelts](num_elements - (num_elements % Self.nelts))

        for i in range(num_elements - (num_elements % Self.nelts), num_elements):
            var flat_index1 = get_broadcast_index(i, tensor1.shape, result_shape)
            var flat_index2 = get_broadcast_index(i, tensor2.shape, result_shape)
            result.store(
                i, func(tensor1.load(flat_index1), tensor2.load(flat_index2))
            )
        return result

    @always_inline("nodebug")
    @staticmethod
    fn tensor_operation[func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],](tensor1: TensorType[dtype], value: Scalar[dtype]) -> TensorType[dtype]:
        """Performs an element-wise operation on tensor with given Scalar."""
        constrained[dtype.is_numeric(), "dtype must be numeric"]()

        var result = TensorType[dtype](tensor1.shape)
        var num_elements = result.shape.num_elements

        @parameter
        fn operation[nelts: Int](idx: Int):
            result.store[nelts](
                idx, func[dtype, nelts](tensor1.load[nelts](idx), value)
            )

        vectorize[operation, Self.nelts](num_elements - (num_elements % Self.nelts))

        for i in range(num_elements - (num_elements % Self.nelts), num_elements):
            result.store[nelts=1](i, func(tensor1.load[nelts=1](i), value))
        return result

    @always_inline("nodebug")
    @staticmethod
    fn tensor_operation[func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts], Int
    ) -> SIMD[dtype, nelts],](tensor1: TensorType[dtype], value: Int) -> TensorType[dtype]:
        """Performs an element-wise operation on tensor with given Int."""
        constrained[dtype.is_numeric(), "dtype must be numeric"]()

        var result = TensorType[dtype](tensor1.shape)
        var num_elements = result.shape.num_elements

        @parameter
        fn operation[nelts: Int](idx: Int):
            result.store[nelts](
                idx, func[dtype, nelts](tensor1.load[nelts](idx), value)
            )

        vectorize[operation, Self.nelts](num_elements - (num_elements % Self.nelts))

        for i in range(num_elements - (num_elements % Self.nelts), num_elements):
            result.store[nelts=1](i, func(tensor1.load[nelts=1](i), value))
        return result

    @always_inline("nodebug")
    @staticmethod
    fn tensor_operation[func: fn[dtype: DType, nelts: Int] (
        SIMD[dtype, nelts]
    ) -> SIMD[dtype, nelts],](tensor1: TensorType[dtype],) -> TensorType[dtype]:
        """Performs an element-wise operation on tensor."""
        constrained[dtype.is_numeric(), "dtype must be numeric"]()

        var result = TensorType[dtype](tensor1.shape)
        var num_elements = result.shape.num_elements

        @parameter
        fn operation[nelts: Int](idx: Int):
            result.store[nelts](
                idx, func[dtype, nelts](tensor1.load[nelts](idx))
            )

        vectorize[operation, Self.nelts](num_elements - (num_elements % Self.nelts))

        for i in range(num_elements - (num_elements % Self.nelts), num_elements):
            result.store[nelts=1](i, func(tensor1.load[nelts=1](i)))
        return result
