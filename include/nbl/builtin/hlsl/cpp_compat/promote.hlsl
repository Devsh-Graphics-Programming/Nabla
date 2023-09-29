#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_PROMOTE_INCLUDED_

namespace nbl
{
namespace hlsl
{

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

#ifdef __HLSL_VERSION

template<typename U>
struct Promote<float1, U>
{
    float3 operator()(U v)
    {
        return float3(v);
    }
};

template<typename U>
struct Promote<float2, U>
{
    float3 operator()(U v)
    {
        return float3(v, v);
    }
};

template<typename U>
struct Promote<float3, U>
{
    float3 operator()(U v)
    {
        return float3(v, v, v);
    }
};

template<typename U>
struct Promote<float4, U>
{
    float3 operator()(U v)
    {
        return float3(v, v, v, v);
    }
};

#endif

}

template<typename T, typename U>
T promote(U v) // TODO: use NBL_CONST_REF_ARG instead of U v
{
    impl::Promote<T, U> _promote;
    return _promote(v);
}

}
}

#endif