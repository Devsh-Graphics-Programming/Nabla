#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_TYPE_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_TYPE_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl
{
namespace hlsl
{
template<typename V>
struct scalar_type
{
    using type = void;
};

template<typename T, uint16_t N>
struct scalar_type<vector<T,N> >
{
    using type = T;
};

template<typename T, uint16_t N, uint16_t M>
struct scalar_type<matrix<T,N,M> >
{
    using type = T;
};
}
}

#endif