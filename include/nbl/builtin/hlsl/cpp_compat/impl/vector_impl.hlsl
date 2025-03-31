#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_VECTOR_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_VECTOR_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

// To prevent implicit truncation warnings
namespace nbl
{
namespace hlsl
{
namespace impl
{

template<typename T, uint16_t N, uint16_t M> NBL_PARTIAL_REQ_TOP(N <= M)
struct static_cast_helper<vector<T, N>, vector<T, M> NBL_PARTIAL_REQ_BOT(N <= M) >
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<T, N> cast(NBL_CONST_REF_ARG(vector<T, M>) val)
    {
        vector<T, N> retVal;
        [[unroll]]
        for (uint16_t i = 0; i < N; i++)
        {
            retVal[i] = val[i];
        }
        return retVal;
    }
};

}
}
}

#endif