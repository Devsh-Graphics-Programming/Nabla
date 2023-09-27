#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_

namespace nbl::hlsl
{
#ifndef __HLSL_VERSION
namespace impl
{
// partial specialize this for `T=matrix<scalar_t,,>|vector<scalar_t,>` and `U=matrix<scalar_t,,>|vector<scalar_t,>|scalar_t`
template<typename T, typename U>
struct Promote
{
    T operator()(U v)
    {
        return T(v);
    }
};
}

template<typename T, typename U>
T promote(U v) // TODO: use NBL_CONST_REF_ARG instead of U v
{
    impl::Promote<T, U> _promote;
    return _promote(v);
}

#endif
}

#endif