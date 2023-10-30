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

template<class T>
struct valid_expression : true_type {};

template<> struct valid_expression<void> : false_type {};

template<class T>
T declval(){}

}

}

}


#define NBL_GENERATE_MEMBER_TESTER(mem) \
namespace nbl \
{ \
namespace hlsl \
{ \
    template<class T, class=bool_constant<true> > \
    struct has_member_##mem : false_type {}; \
    template<class T> \
    struct has_member_##mem<T,bool_constant<impl::valid_expression<__decltype(impl::declval<T>().mem)>::value> > : true_type {}; \
} \
}


NBL_GENERATE_MEMBER_TESTER(x)
NBL_GENERATE_MEMBER_TESTER(y)
NBL_GENERATE_MEMBER_TESTER(z)
NBL_GENERATE_MEMBER_TESTER(w)

#endif
#endif