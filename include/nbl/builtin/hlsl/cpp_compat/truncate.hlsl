#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_TRUNCATE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_TRUNCATE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif

namespace nbl
{
namespace hlsl
{

namespace concepts
{
  template<typename To, typename From>
  NBL_BOOL_CONCEPT can_truncate_vector = concepts::Vectorial<To> && concepts::Vectorial<From> && concepts::same_as<typename vector_traits<To>::scalar_type, typename vector_traits<From>::scalar_type > && vector_traits<To>::Dimension <= vector_traits<From>::Dimension;
}

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

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::can_truncate_vector<To, From>) 
struct Truncate<To, From NBL_PARTIAL_REQ_BOT(concepts::can_truncate_vector<To, From>) >
{
    NBL_CONSTEXPR_FUNC To operator()(const From v)
    {
        array_get<From, typename vector_traits<From>::scalar_type> getter;
        array_set<To, typename vector_traits<To>::scalar_type> setter;
        To output;
        [[unroll]]
        for (int i = 0; i < vector_traits<To>::Dimension; ++i)
            setter(output, i, getter(v, i));
        return output;
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