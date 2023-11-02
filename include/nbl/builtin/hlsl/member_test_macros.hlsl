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

enum e_member_presence
{
    absent = 0,
    non_static,
    as_static
};

template<class T>
T declval(){}

}

typedef impl::e_member_presence e_member_presence;
}

}


#define NBL_GENERATE_MEMBER_TESTER(mem) \
namespace nbl \
{ \
namespace hlsl \
{ \
namespace impl \
{ \
template<class T, class=void> \
struct is_static_member_##mem: false_type {}; \
template<class T> \
struct is_static_member_##mem<T,typename enable_if<!is_same<decltype(T::mem),void>::value,void>::type>: true_type{}; \
template<class T, class=void> \
struct is_member_##mem: false_type {}; \
template<class T> \
struct is_member_##mem<T,typename enable_if<!is_same<decltype(impl::declval<T>().mem),void>::value,void>::type>: true_type{}; \
} \
template<class T> \
struct has_member_##mem \
{ \
    NBL_CONSTEXPR_STATIC_INLINE e_member_presence value = (e_member_presence)(impl::is_member_##mem<T>::value + impl::is_static_member_##mem<T>::value); \
}; \
} \
}


NBL_GENERATE_MEMBER_TESTER(x)
NBL_GENERATE_MEMBER_TESTER(y)
NBL_GENERATE_MEMBER_TESTER(z)
NBL_GENERATE_MEMBER_TESTER(w)

#endif
#endif