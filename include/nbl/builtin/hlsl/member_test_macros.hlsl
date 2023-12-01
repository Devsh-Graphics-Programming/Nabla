// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MEMBER_TEST_MACROS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMBER_TEST_MACROS_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>

#ifdef __HLSL_VERSION

namespace nbl
{
namespace hlsl
{

namespace impl
{

template<class T, bool C>
struct is_const_helper : bool_constant<C>
{
    NBL_CONSTEXPR_STATIC_INLINE bool is_constant = is_const<T>::value;
};

enum e_member_presence
{
    absent = 0,
    non_static = 1,
    as_static = 2,
    static_constexpr = 3,
};

template<class T>
T declval(){}

}

typedef impl::e_member_presence e_member_presence;
}

}


#define NBL_GENERATE_MEMBER_TESTER(a) \
namespace nbl \
{ \
namespace hlsl \
{ \
namespace impl { \
template<class T, class=void>  \
struct is_static_member_##a: false_type {NBL_CONSTEXPR_STATIC_INLINE bool is_constant = false; }; \
template<class T>  \
struct is_static_member_##a<T,typename enable_if<!is_same<decltype(T::a),void>::value,void>::type>: is_const_helper<decltype(T::a), true> {}; \
template<class T, class=void> \
struct is_member_##a: false_type {NBL_CONSTEXPR_STATIC_INLINE bool is_constant = false;}; \
template<class T> \
struct is_member_##a<T,typename enable_if<!is_same<decltype(declval<T>().a),void>::value,void>::type> : is_const_helper<decltype(declval<T>().a), true>{}; \
} \
template<class T> \
struct has_member_##a {  NBL_CONSTEXPR_STATIC_INLINE e_member_presence value = (e_member_presence)(impl::is_member_##a<T>::value + impl::is_static_member_##a<T>::value + impl::is_static_member_##a<T>::is_constant); }; \
} \
}


NBL_GENERATE_MEMBER_TESTER(x)
NBL_GENERATE_MEMBER_TESTER(y)
NBL_GENERATE_MEMBER_TESTER(z)
NBL_GENERATE_MEMBER_TESTER(w)

#endif
#endif