#ifndef _NBL_BUILTIN_HLSL_PORTABLE_VECTOR_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_VECTOR_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/emulated/vector_t.hlsl>
#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T, uint32_t N, bool fundamental = is_fundamental<T>::value>
struct portable_vector
{
    using type = vector<T, N>;
};
#ifdef __HLSL_VERSION
template<typename T, uint32_t N>
struct portable_vector<T, N, false>
{
    using type = portable_vector<T, N>;
};
#endif

template<typename T, uint32_t N>
using portable_vector_t = typename portable_vector<T, N>::type;

template<typename T>
using portable_vector_t2 = portable_vector_t<T, 2>;
template<typename T>
using portable_vector_t3 = portable_vector_t<T, 3>;
template<typename T>
using portable_vector_t4 = portable_vector_t<T, 4>;

#ifdef __HLSL_VERSION
template<typename device_caps = void>
using portable_vector64_t2 = portable_vector_t2<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_vector64_t3 = portable_vector_t3<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_vector64_t4 = portable_vector_t4<portable_float64_t<device_caps> >;
#else
template<typename device_caps = void>
using portable_vector64_t2 = portable_vector_t2<float64_t>;
template<typename device_caps = void>
using portable_vector64_t3 = portable_vector_t3<float64_t>;
template<typename device_caps = void>
using portable_vector64_t4 = portable_vector_t4<float64_t>;
#endif

}
}

#endif