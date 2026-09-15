#ifndef _NBL_BUILTIN_HLSL_ARRAY_ACCESSORS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_ARRAY_ACCESSORS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

namespace nbl
{
namespace hlsl
{
template<typename ArrayType, typename ComponentType, typename I = uint32_t>
struct array_get
{
    ComponentType operator()(NBL_CONST_REF_ARG(ArrayType) arr, const I ix) NBL_CONST_MEMBER_FUNC
    {
        return arr[ix];
    }
};

template<typename ArrayType, typename ComponentType, typename I = uint32_t>
struct array_set
{
    void operator()(NBL_REF_ARG(ArrayType) arr, I index, ComponentType val) NBL_CONST_MEMBER_FUNC
    {
        arr[index] = val;
    }
};

namespace impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct MatrixComponentSetterHelper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct MatrixComponentGetterHelper;

// this concept will check whether MatrixType is a matrix and whether its component is a native type scalar, should work only for HLSL matrices
#define MATRIX_COMPONENT_IS_NATIVE_TYPE nbl::hlsl::matrix_traits<MatrixType>::IsMatrix && nbl::hlsl::is_scalar_v<typename nbl::hlsl::matrix_traits<MatrixType>::scalar_type>

template<typename MatrixType>
NBL_PARTIAL_REQ_TOP(MATRIX_COMPONENT_IS_NATIVE_TYPE)
struct MatrixComponentSetterHelper<MatrixType NBL_PARTIAL_REQ_BOT(MATRIX_COMPONENT_IS_NATIVE_TYPE) >
{
    using ComponentType = typename nbl::hlsl::matrix_traits<MatrixType>::scalar_type;
    static void __call(NBL_REF_ARG(MatrixType) mat, uint16_t row, uint16_t column, ComponentType value)
    {
        mat[row][column] = value;
    }
};
template<typename MatrixType>
NBL_PARTIAL_REQ_TOP(MATRIX_COMPONENT_IS_NATIVE_TYPE)
struct MatrixComponentGetterHelper<MatrixType NBL_PARTIAL_REQ_BOT(MATRIX_COMPONENT_IS_NATIVE_TYPE) >
{
    using ComponentType = typename nbl::hlsl::matrix_traits<MatrixType>::scalar_type;
    static ComponentType __call(NBL_REF_ARG(MatrixType) mat, uint16_t row, uint16_t column)
    {
        return mat[row][column];
    }
};

#undef MATRIX_COMPONENT_IS_NATIVE_TYPE

}

template<typename MatrixType>
void matrix_component_set(NBL_REF_ARG(MatrixType) mat, uint16_t row, uint16_t column, typename nbl::hlsl::matrix_traits<MatrixType>::scalar_type value)
{
    impl::MatrixComponentSetterHelper<MatrixType>::__call(mat, row, column, value);
}

template<typename MatrixType>
typename nbl::hlsl::matrix_traits<MatrixType>::scalar_type matrix_component_get(NBL_REF_ARG(MatrixType) mat, uint16_t row, uint16_t column)
{
    return impl::MatrixComponentGetterHelper<MatrixType>::__call(mat, row, column);
}

}
}

#endif
