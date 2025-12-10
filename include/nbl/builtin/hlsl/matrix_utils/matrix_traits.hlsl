#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T>
struct matrix_traits
{
    using scalar_type = T;
    using row_type = void;
    using transposed_type = void;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = 1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = 1;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = false;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMatrix = false;
};

// TODO: when this bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007 is fixed, uncomment and delete template specializations
template<typename T, uint32_t N, uint32_t M>
struct matrix_traits<matrix<T,N,M> >
{
    using scalar_type = T;
    using row_type = vector<T, M>;
    using transposed_type = matrix<T, M, N>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = N;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = M;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMatrix = true;
};

}
}

#endif