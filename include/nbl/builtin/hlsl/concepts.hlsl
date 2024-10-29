// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>


#ifndef __HLSL_VERSION

// TODO: old stuff, see how much we can remove
#define NBL_CONCEPT_TYPE_PARAMS(...) template <__VA_ARGS__>
#define NBL_CONCEPT_SIGNATURE(NAME, ...) concept NAME = requires(__VA_ARGS__)
#define NBL_CONCEPT_BODY(...) { __VA_ARGS__ };
#define NBL_CONCEPT_ASSIGN(NAME, ...) concept NAME = __VA_ARGS__;
#define NBL_REQUIRES(...) requires __VA_ARGS__ 

// for struct definitions, use instead of closing `>` on the primary template parameter list
#define NBL_PRIMARY_REQUIRES(...) > requires (__VA_ARGS__)

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE
// NOTE: C++20 requires and C++11 enable_if have to be in different places! ITS OF UTTMOST IMPORTANCE YOUR REQUIRE CLAUSES ARE IDENTICAL FOR BOTH MACROS
// put just after the closing `>` on the partial template specialization `template` declaration e.g. `template<typename U, typename V, typename T> NBL_PARTIAL_REQ_TOP(SomeCond<U>)
#define NBL_PARTIAL_REQ_TOP(...) requires (__VA_ARGS__)
// put just before closing `>` on the partial template specialization Type args, e.g. `MyStruct<U,V,T NBL_PARTIAL_REQ_BOT(SomeCond<U>)>
#define NBL_PARTIAL_REQ_BOT(...)

// condition
#define NBL_FUNC_REQUIRES_BEGIN(...) requires (__VA_ARGS__)
// return value
#define NBL_FUNC_REQUIRES_END(...) __VA_ARGS__

#include <concepts>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

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

}
}
}

#else

// TODO: old stuff, see how much we can remove
// No C++20 support. Do nothing.
#define NBL_CONCEPT_TYPE_PARAMS(...)
#define NBL_CONCEPT_SIGNATURE(NAME, ...) 
#define NBL_CONCEPT_BODY(...)
#define NBL_REQUIRES(...)


// for struct definitions, use instead of closing `>` on the primary template parameter list
#define NBL_PRIMARY_REQUIRES(...) ,typename __requires=::nbl::hlsl::enable_if_t<(__VA_ARGS__),void> > 

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE ,typename __requires=void
// NOTE: C++20 requires and C++11 enable_if have to be in different places! ITS OF UTTMOST IMPORTANCE YOUR REQUIRE CLAUSES ARE IDENTICAL FOR BOTH MACROS
// put just after the closing `>` on the partial template specialization `template` declaration e.g. `template<typename U, typename V, typename T> NBL_PARTIAL_REQ_TOP(SomeCond<U>)
#define NBL_PARTIAL_REQ_TOP(...)
// put just before closing `>` on the partial template specialization Type args, e.g. `MyStruct<U,V,T NBL_PARTIAL_REQ_BOT(SomeCond<U>)>
#define NBL_PARTIAL_REQ_BOT(...) ,std::enable_if_t<(__VA_ARGS__),void> 

// condition, use right after the closing `>` of a function template
#define NBL_FUNC_REQUIRES_BEGIN(...) ::nbl::hlsl::enable_if_t<(__VA_ARGS__),
// return value, use `END(T)` instead of the return value type declaration
#define NBL_FUNC_REQUIRES_END(...) __VA_ARGS__> 

#endif


#endif