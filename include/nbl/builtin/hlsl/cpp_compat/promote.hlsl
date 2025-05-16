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
template<typename T, typename U>
struct Promote
{
    NBL_CONSTEXPR_FUNC T operator()(NBL_CONST_REF_ARG(U) v)
    {
        return T(v);
    }
};

#ifdef __HLSL_VERSION

template<typename Scalar, typename U>
struct Promote<vector <Scalar, 1>, U>
{
    NBL_CONSTEXPR_FUNC enable_if_t<is_scalar<Scalar>::value && is_scalar<U>::value, vector <Scalar, 1> > operator()(NBL_CONST_REF_ARG(U) v)
    {
        vector <Scalar, 1> promoted = {Scalar(v)};
        return promoted;
    }
};

template<typename Scalar, typename U>
struct Promote<vector <Scalar, 2>, U>
{
    NBL_CONSTEXPR_FUNC enable_if_t<is_scalar<Scalar>::value && is_scalar<U>::value, vector <Scalar, 2> > operator()(NBL_CONST_REF_ARG(U) v)
    {
        vector <Scalar, 2> promoted = {Scalar(v), Scalar(v)};
        return promoted;
    }
};

template<typename Scalar, typename U>
struct Promote<vector <Scalar, 3>, U>
{
    NBL_CONSTEXPR_FUNC enable_if_t<is_scalar<Scalar>::value && is_scalar<U>::value, vector <Scalar, 3> > operator()(NBL_CONST_REF_ARG(U) v)
    {
        vector <Scalar, 3> promoted = {Scalar(v), Scalar(v), Scalar(v)};
        return promoted;
    }
};

template<typename Scalar, typename U>
struct Promote<vector <Scalar, 4>, U>
{
    NBL_CONSTEXPR_FUNC enable_if_t<is_scalar<Scalar>::value && is_scalar<U>::value, vector <Scalar, 4> > operator()(NBL_CONST_REF_ARG(U) v)
    {
        vector <Scalar, 4> promoted = {Scalar(v), Scalar(v), Scalar(v), Scalar(v)};
        return promoted;
    }
};

#endif

}

template<typename T, typename U>
NBL_CONSTEXPR_FUNC T promote(const U v) // TODO: use NBL_CONST_REF_ARG(U) instead of U v (circular ref)
{
    impl::Promote<T,U> _promote;
    return _promote(v);
}

}
}

#endif