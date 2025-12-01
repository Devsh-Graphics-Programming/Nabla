#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace impl
{

// partial specialize this for `T=matrix<scalar_t,,>|vector<scalar_t,>` and `U=matrix<scalar_t,,>|vector<scalar_t,>|scalar_t`
template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct Promote
{
    NBL_CONSTEXPR_FUNC T operator()(NBL_CONST_REF_ARG(U) v)
    {
        return T(v);
    }
};

// TODO(kevinyu): Should we enable truncation from uint64_t to emulated_vector<emulated_uint64_t, N>?
template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::Vectorial<To> && is_scalar_v<From> && is_same_v<typename vector_traits<To>::scalar_type, From>)
struct Promote<To, From NBL_PARTIAL_REQ_BOT(concepts::Vectorial<To> && is_scalar_v<From> && is_same_v<typename vector_traits<To>::scalar_type, From>) >
{
    NBL_CONSTEXPR_FUNC To operator()(const From v)
    {
        array_set<To, From> setter;
        To output;
        [[unroll]]
        for (int i = 0; i < vector_traits<To>::Dimension; ++i)
            setter(output, i, v);
        return output;
    }
};

}

template<typename T, typename U>
NBL_CONSTEXPR_FUNC T promote(NBL_CONST_REF_ARG(U) v)
{
    impl::Promote<T,U> _promote;
    return _promote(v);
}


}
}

#endif