// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MEMBER_TEST_MACROS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMBER_TEST_MACROS_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <boost/preprocessor.hpp>

#ifdef __HLSL_VERSION

namespace nbl
{
namespace hlsl
{

namespace impl
{

enum e_member_presence
{
    is_present = 1<<0,
    is_static  = 1<<1,
    is_const   = 1<<2,
};

template<class T>
T declval(){}

template<bool=false>
struct if_2_else_1 : integral_constant<uint32_t,1> {};
template<>
struct if_2_else_1<true> : integral_constant<uint32_t,2> {};

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
struct is_static_member_##a: false_type { }; \
template<class T>  \
struct is_static_member_##a<T,typename enable_if<!is_same<decltype(T::a),void>::value,void>::type> : true_type {  }; \
template<class T, class=void> \
struct is_member_##a: false_type { using type = void; }; \
template<class T> \
struct is_member_##a<T,typename enable_if<!is_same<decltype(declval<T>().a),void>::value,void>::type> : true_type { using type = decltype(declval<T>().a); }; \
} \
template<class T> \
struct has_member_##a {  NBL_CONSTEXPR_STATIC_INLINE e_member_presence value = (e_member_presence)(impl::is_member_##a<T>::value + 2*impl::is_static_member_##a<T>::value + 4*is_const<typename impl::is_member_##a<T>::type>::value); }; \
template<class T, class F> struct has_member_##a##_with_type : bool_constant<has_member_##a<T>::value && is_same<typename impl::is_member_##a<T>::type, F>::value> {}; \
} \
}


NBL_GENERATE_MEMBER_TESTER(x)
NBL_GENERATE_MEMBER_TESTER(y)
NBL_GENERATE_MEMBER_TESTER(z)
NBL_GENERATE_MEMBER_TESTER(w)



#define NBL_TYPE_DECLARE(z, n, x) BOOST_PP_COMMA_IF(x) typename Arg##n
#define NBL_TYPE_DECLARE_DEFAULT(z, n, x) BOOST_PP_COMMA_IF(x) typename Arg##n=void
#define NBL_TYPE_FWD(z, n, x) BOOST_PP_COMMA_IF(x) Arg##n
#define NBL_DECLVAL_DECLARE(z, n, x) impl::declval<Arg##n>() BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(BOOST_PP_INC(n), x))

#define GENERATE_STATIC_METHOD_TESTER_SPEC(z, n, x) \
template<class T BOOST_PP_REPEAT(n, NBL_TYPE_DECLARE, n)> \
struct has_static_method_##x<T BOOST_PP_REPEAT(n, NBL_TYPE_FWD, n), typename make_void<decltype(T::x(BOOST_PP_REPEAT(n, NBL_DECLVAL_DECLARE, n)))>::type> : true_type \
{ \
    using return_type = decltype(T::x(BOOST_PP_REPEAT(n, NBL_DECLVAL_DECLARE, n))); \
    NBL_CONSTEXPR_STATIC_INLINE uint arg_count = n; \
}; 

#define GENERATE_STATIC_METHOD_TESTER(x, n) \
template<typename T BOOST_PP_REPEAT(n, NBL_TYPE_DECLARE_DEFAULT, n), class=void> \
struct has_static_method_##x : false_type {}; \
BOOST_PP_REPEAT(n, GENERATE_STATIC_METHOD_TESTER_SPEC, x)

#define GENERATE_METHOD_TESTER_SPEC(z, n, x) \
template<class T BOOST_PP_REPEAT(n, NBL_TYPE_DECLARE, n)> \
struct has_method_##x<T BOOST_PP_REPEAT(n, NBL_TYPE_FWD, n), typename make_void<decltype(impl::declval<T>().x(BOOST_PP_REPEAT(n, NBL_DECLVAL_DECLARE, n)))>::type> : impl::if_2_else_1<impl::has_static_method_##x<T BOOST_PP_REPEAT(n, NBL_TYPE_FWD, n)>::value> \
{ \
    using return_type = decltype(impl::declval<T>().x(BOOST_PP_REPEAT(n, NBL_DECLVAL_DECLARE, n))); \
    NBL_CONSTEXPR_STATIC_INLINE uint arg_count = n; \
}; 

/* 
    types that are impilicitly convertible to each other mess this check up 

    struct S
    {
        void a(int) { return 0;}
    };

    has_method_a<S, float>::value will be true 
    since float is implicitly convertible to int and
    due to how we check function signatures at the moment
*/

#define GENERATE_METHOD_TESTER(x) \
namespace nbl { \
namespace hlsl { \
namespace impl { GENERATE_STATIC_METHOD_TESTER(x, 4) } \
template<typename T BOOST_PP_REPEAT(4, NBL_TYPE_DECLARE_DEFAULT, 4), class=void> \
struct has_method_##x : false_type {}; \
BOOST_PP_REPEAT(4, GENERATE_METHOD_TESTER_SPEC, x) \
}}


GENERATE_METHOD_TESTER(a)
GENERATE_METHOD_TESTER(b)


#endif
#endif