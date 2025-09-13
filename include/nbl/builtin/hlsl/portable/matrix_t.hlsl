#ifndef _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/emulated/matrix_t.hlsl>
#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T, uint32_t N, uint32_t M, bool fundamental = is_fundamental<T>::value>
struct portable_matrix
{
    using type = matrix<T, N, M>;
};
#ifdef __HLSL_VERSION
template<typename T, uint32_t N, uint32_t M>
struct portable_matrix<T, N, M, false>
{
    using type = emulated_matrix<T, N, M>;
};
#endif

template<typename T, uint32_t N, uint32_t M>
using portable_matrix_t = typename portable_matrix<T, N, M>::type;

template<typename T>
using portable_matrix_t2x2 = portable_matrix_t<T, 2, 2>;
template<typename T>
using portable_matrix_t2x3 = portable_matrix_t<T, 2, 3>;
template<typename T>
using portable_matrix_t2x4 = portable_matrix_t<T, 2, 4>;
template<typename T>
using portable_matrix_t3x2 = portable_matrix_t<T, 3, 2>;
template<typename T>
using portable_matrix_t3x3 = portable_matrix_t<T, 3, 3>;
template<typename T>
using portable_matrix_t3x4 = portable_matrix_t<T, 3, 4>;
template<typename T>
using portable_matrix_t4x2 = portable_matrix_t<T, 4, 2>;
template<typename T>
using portable_matrix_t4x3 = portable_matrix_t<T, 4, 3>;
template<typename T>
using portable_matrix_t4x4 = portable_matrix_t<T, 4, 4>;

}
}

#endif