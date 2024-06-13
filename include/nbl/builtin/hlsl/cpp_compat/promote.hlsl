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

// TODO: write this scalar-type agnostic!
template<typename U>
struct Promote<float32_t1,U>
{
    float32_t1 operator()(U v)
    {
        return float32_t1(v);
    }
};

template<typename U>
struct Promote<float32_t2, U>
{
    float32_t2 operator()(U v)
    {
        return float32_t2(v, v);
    }
};

template<typename U>
struct Promote<float32_t3, U>
{
    float32_t3 operator()(U v)
    {
        return float32_t3(v, v, v);
    }
};

template<typename U>
struct Promote<float32_t4, U>
{
    float32_t4 operator()(U v)
    {
        return float32_t4(v, v, v, v);
    }
};

template<typename U>
struct Promote<float64_t1,U>
{
    float64_t1 operator()(U v)
    {
        return float64_t1(v);
    }
};

template<typename U>
struct Promote<float64_t2, U>
{
    float64_t2 operator()(U v)
    {
        return float64_t2(v, v);
    }
};

template<typename U>
struct Promote<float64_t3, U>
{
    float64_t3 operator()(U v)
    {
        return float64_t3(v, v, v);
    }
};

template<typename U>
struct Promote<float64_t4, U>
{
    float64_t4 operator()(U v)
    {
        return float64_t4(v, v, v, v);
    }
};

#endif

}

template<typename T, typename U>
T promote(const U v) // TODO: use NBL_CONST_REF_ARG(U) instead of U v (circular ref)
{
    impl::Promote<T,U> _promote;
    return _promote(v);
}

}
}

#endif