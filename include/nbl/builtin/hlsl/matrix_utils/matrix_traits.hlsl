#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/quaternion/quaternion.hlsl>
// TODO: remove this header when deleting vectorSIMDf.hlsl
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"
#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename MatT>
struct matrix_traits;
// partial spec for matrices
template<typename T, int32_t N, int32_t M>
struct matrix_traits<matrix<T, N, M> >
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = N;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = M;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = N == M;
};

}
}

#endif