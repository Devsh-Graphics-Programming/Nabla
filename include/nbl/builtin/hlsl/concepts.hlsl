// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/utility.hlsl>


namespace nbl
{
namespace hlsl
{
namespace concepts
{
// common implementation juice
#include <boost/preprocessor/seq/elem.hpp>
#define NBL_IMPL_CONCEPT_FULL_TPLT(z, n, unused) BOOST_PP_SEQ_ELEM(n,NBL_CONCEPT_TPLT_PRM_KINDS) BOOST_PP_SEQ_ELEM(n,NBL_CONCEPT_TPLT_PRM_NAMES)
#include <boost/preprocessor/repetition/enum.hpp>
#define NBL_CONCEPT_FULL_TPLT() BOOST_PP_ENUM(BOOST_PP_SEQ_SIZE(NBL_CONCEPT_TPLT_PRM_NAMES),NBL_IMPL_CONCEPT_FULL_TPLT,DUMMY)
#include <boost/preprocessor/seq/enum.hpp>
#define NBL_CONCEPT_TPLT_PARAMS() BOOST_PP_SEQ_ENUM(NBL_CONCEPT_TPLT_PRM_NAMES)
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
//
#define NBL_CONCEPT_REQ_TYPE 0
#define NBL_CONCEPT_REQ_EXPR 1
//
#define NBL_CONCEPT_REQ_EXPR_RET_TYPE 2


//! Now diverge
#ifndef __cpp_concepts


// to define a concept using `concept Name = SomeContexprBoolCondition<T>;`
#define NBL_BOOL_CONCEPT concept

// for struct definitions, use instead of closing `>` on the primary template parameter list
#define NBL_PRIMARY_REQUIRES(...) > requires (__VA_ARGS__)

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE
// NOTE: C++20 requires and C++11 enable_if have to be in different places! ITS OF UTTMOST IMPORTANCE YOUR REQUIRE CLAUSES ARE IDENTICAL FOR BOTH MACROS
// put just after the closing `>` on the partial template specialization `template` declaration e.g. `template<typename U, typename V, typename T> NBL_PARTIAL_REQ_TOP(SomeCond<U>)
#define NBL_PARTIAL_REQ_TOP(...) requires (__VA_ARGS__)
// put just before closing `>` on the partial template specialization Type args, e.g. `MyStruct<U,V,T NBL_PARTIAL_REQ_BOT(SomeCond<U>)>
#define NBL_PARTIAL_REQ_BOT(...)

// condition, use instead of the closing `>` of a function template
#define NBL_FUNC_REQUIRES(...) > requires (__VA_ARGS__)


//
#define NBL_CONCEPT_PARAM_T(ID,...) ID
//
#define NBL_IMPL_IMPL_CONCEPT_BEGIN(A,...) __VA_ARGS__ A
#define NBL_IMPL_CONCEPT_BEGIN(z,n,data) NBL_IMPL_IMPL_CONCEPT_BEGIN NBL_CONCEPT_PARAM_##n
// TODO: are empty local parameter lists valid? a.k.a. just a `()`
#define NBL_CONCEPT_BEGIN(LOCAL_PARAM_COUNT) template<NBL_CONCEPT_FULL_TPLT()> \
concept NBL_CONCEPT_NAME = requires BOOST_PP_EXPR_IF(LOCAL_PARAM_COUNT,(BOOST_PP_ENUM(LOCAL_PARAM_COUNT,NBL_IMPL_CONCEPT_BEGIN,DUMMY))) \
{
//
#define NBL_IMPL_CONCEPT_REQ_TYPE(...) typename __VA_ARGS__;
#define NBL_IMPL_CONCEPT_REQ_EXPR(...) __VA_ARGS__;
#define NBL_IMPL_CONCEPT_REQ_EXPR_RET_TYPE(E,C,...) {E}; C<decltype E,__VA_ARGS__ >;
//
#define NBL_IMPL_CONCEPT (NBL_IMPL_CONCEPT_REQ_TYPE,NBL_IMPL_CONCEPT_REQ_EXPR,NBL_IMPL_CONCEPT_REQ_EXPR_RET_TYPE)
//
#define NBL_IMPL_CONCEPT_END_DEF(r,unused,i,e) NBL_EVAL(BOOST_PP_TUPLE_ELEM(BOOST_PP_SEQ_HEAD(e),NBL_IMPL_CONCEPT) BOOST_PP_SEQ_TAIL(e))
//
#define NBL_CONCEPT_END(SEQ) BOOST_PP_SEQ_FOR_EACH_I(NBL_IMPL_CONCEPT_END_DEF, DUMMY, SEQ) \
}


#include <concepts>

// Alias some of the std concepts in nbl. As this is C++20 only, we don't need to use
// the macros here.
template <typename T, typename U>
concept same_as = std::same_as<T, U>;

template <typename D, typename B>
concept derived_from = std::derived_from<D, B>;

template <typename F, typename T>
concept convertible_to = std::convertible_to<F, T>;

template <typename T, typename F>
concept assignable_from = std::assignable_from<T, F>;

template <typename T, typename U>
concept common_with = std::common_with<T, U>;

template <typename T>
concept integral = std::integral<T>;

template <typename T>
concept signed_integral = std::signed_integral<T>;

template <typename T>
concept unsigned_integral = std::unsigned_integral<T>;

template <typename T>
concept floating_point = std::floating_point<T>;


// Some other useful concepts.

template<typename T, typename... Ts>
concept any_of = (same_as<T, Ts> || ...);

template <typename T>
concept scalar = floating_point<T> || integral<T>;

template <typename T>
concept vectorial = is_vector<T>::value;

template <typename T>
concept matricial = is_matrix<T>::value;

#else


// to define a concept using `concept Name = SomeContexprBoolCondition<T>;`
#define NBL_BOOL_CONCEPT NBL_CONSTEXPR bool

// for struct definitions, use instead of closing `>` on the primary template parameter list
#define NBL_PRIMARY_REQUIRES(...) ,typename __requires=::nbl::hlsl::enable_if_t<(__VA_ARGS__),void> > 

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE ,typename __requires=void
// NOTE: C++20 requires and C++11 enable_if have to be in different places! ITS OF UTTMOST IMPORTANCE YOUR REQUIRE CLAUSES ARE IDENTICAL FOR BOTH MACROS
// put just after the closing `>` on the partial template specialization `template` declaration e.g. `template<typename U, typename V, typename T> NBL_PARTIAL_REQ_TOP(SomeCond<U>)
#define NBL_PARTIAL_REQ_TOP(...)
// put just before closing `>` on the partial template specialization Type args, e.g. `MyStruct<U,V,T NBL_PARTIAL_REQ_BOT(SomeCond<U>)>
#define NBL_PARTIAL_REQ_BOT(...) ,std::enable_if_t<(__VA_ARGS__),void> 

// condition, use instead of the closing `>` of a function template
#define NBL_FUNC_REQUIRES(...) ,std::enable_if_t<(__VA_ARGS__),bool> = true>


//
#define NBL_CONCEPT_BEGIN(LOCAL_PARAM_COUNT) namespace BOOST_PP_CAT(__concept__,NBL_CONCEPT_NAME) \
{
//
#define NBL_CONCEPT_PARAM_T(ID,...) ::nbl::hlsl::experimental::declval<__VA_ARGS__ >()
//
#define NBL_IMPL_CONCEPT_REQ_TYPE(...) ::nbl::hlsl::make_void_t<typename __VA_ARGS__ >
#define NBL_IMPL_CONCEPT_REQ_EXPR(...) ::nbl::hlsl::make_void_t<decltype(__VA_ARGS__)>
#define NBL_IMPL_CONCEPT_REQ_EXPR_RET_TYPE(E,C,...) ::nbl::hlsl::enable_if_t<C<decltype E ,__VA_ARGS__  > >
//
#define NBL_IMPL_CONCEPT_SFINAE (NBL_IMPL_CONCEPT_REQ_TYPE,NBL_IMPL_CONCEPT_REQ_EXPR,NBL_IMPL_CONCEPT_REQ_EXPR_RET_TYPE)
//
#define NBL_IMPL_CONCEPT_END_DEF(r,unused,i,e) template<NBL_CONCEPT_FULL_TPLT(), typename=void> \
struct BOOST_PP_CAT(__requirement,i) : ::nbl::hlsl::false_type {}; \
template<NBL_CONCEPT_FULL_TPLT()> \
struct BOOST_PP_CAT(__requirement,i)<NBL_CONCEPT_TPLT_PARAMS(), \
NBL_EVAL(BOOST_PP_TUPLE_ELEM(BOOST_PP_SEQ_HEAD(e),NBL_IMPL_CONCEPT_SFINAE) BOOST_PP_SEQ_TAIL(e)) \
 > : ::nbl::hlsl::true_type {};
//
#define NBL_IMPL_CONCEPT_END_GET(r,unused,i,e) BOOST_PP_EXPR_IF(i,&&) BOOST_PP_CAT(__concept__,NBL_CONCEPT_NAME)::BOOST_PP_CAT(__requirement,i)<NBL_CONCEPT_TPLT_PARAMS()>::value
//
#define NBL_CONCEPT_END(SEQ) BOOST_PP_SEQ_FOR_EACH_I(NBL_IMPL_CONCEPT_END_DEF, DUMMY, SEQ) \
} \
template<NBL_CONCEPT_FULL_TPLT()> \
NBL_CONSTEXPR bool NBL_CONCEPT_NAME = BOOST_PP_SEQ_FOR_EACH_I(NBL_IMPL_CONCEPT_END_GET, DUMMY, SEQ)

// TODO: counterparts of all the other concepts

#endif
}
}
}

#endif