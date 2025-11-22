#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_TRUNCATE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_TRUNCATE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"

namespace nbl
{
namespace hlsl
{

namespace impl
{

template<typename T, typename U NBL_STRUCT_CONSTRAINABLE >
struct Truncate
{
    NBL_CONSTEXPR_FUNC T operator()(NBL_CONST_REF_ARG(U) v)
    {
        return T(v);
    }
};

template<typename Scalar, uint16_t N> NBL_PARTIAL_REQ_TOP(concepts::Scalar<Scalar>)
struct Truncate<vector<Scalar, 1>, vector<Scalar, N> NBL_PARTIAL_REQ_BOT(concepts::Scalar<Scalar>) >
{
    NBL_CONSTEXPR_FUNC vector<Scalar, 1> operator()(NBL_CONST_REF_ARG(vector<Scalar, N>) v)
    {
        vector<Scalar, 1> truncated = { v[0] };
        return truncated;
    }
};

template<typename Scalar, uint16_t N> NBL_PARTIAL_REQ_TOP(concepts::Scalar<Scalar> && N >= 2)
struct Truncate<vector<Scalar, 2>, vector<Scalar, N> NBL_PARTIAL_REQ_BOT(concepts::Scalar<Scalar> && N >= 2) >
{
    NBL_CONSTEXPR_FUNC vector<Scalar, 2> operator()(NBL_CONST_REF_ARG(vector<Scalar, N>) v)
    {
        vector<Scalar, 2> truncated = { v[0], v[1]};
        return truncated;
    }
};

template<typename Scalar, uint16_t N> NBL_PARTIAL_REQ_TOP(concepts::Scalar<Scalar>&& N >= 3)
struct Truncate<vector<Scalar, 3>, vector<Scalar, N> NBL_PARTIAL_REQ_BOT(concepts::Scalar<Scalar>&& N >= 3) >
{
    NBL_CONSTEXPR_FUNC vector<Scalar, 3> operator()(NBL_CONST_REF_ARG(vector<Scalar, N>) v)
    {
        vector<Scalar, 3> truncated = { v[0], v[1], v[2] };
        return truncated;
    }
};

template<typename Scalar, uint16_t N> NBL_PARTIAL_REQ_TOP(concepts::Scalar<Scalar>&& N >= 4)
struct Truncate<vector<Scalar, 4>, vector<Scalar, N> NBL_PARTIAL_REQ_BOT(concepts::Scalar<Scalar>&& N >= 4) >
{
    NBL_CONSTEXPR_FUNC vector<Scalar, 4> operator()(NBL_CONST_REF_ARG(vector<Scalar, N>) v)
    {
        vector<Scalar, 4> truncated = { v[0], v[1], v[2], v[3] };
        return truncated;
    }
};

} //namespace impl

template<typename T, typename U>
NBL_CONSTEXPR_FUNC T truncate(NBL_CONST_REF_ARG(U) v)
{
    impl::Truncate<T, U> _truncate;
    return _truncate(v);
}

}
}

#endif