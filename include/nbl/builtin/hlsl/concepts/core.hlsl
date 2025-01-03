// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_CORE_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_CORE_HLSL_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

#ifdef __cpp_concepts // CPP

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

#elif defined(__HLSL_VERSION) // HLSL

template<typename T, typename U>
NBL_BOOL_CONCEPT same_as = is_same_v<T, U>;

// TODO: implement when hlsl::is_base_of is done
//#define NBL_CONCEPT_NAME derived_from
// ...

// TODO: implement when hlsl::is_converible is done
//#define NBL_CONCEPT_NAME convertible_to
// ...

// TODO?
//#define NBL_CONCEPT_NAME assignable_from

// TODO?
//template <typename T, typename U>
//concept common_with = std::common_with<T, U>;

template<typename T>
NBL_BOOL_CONCEPT integral = nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT signed_integral = nbl::hlsl::is_signed_v<T> && nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT unsigned_integral = !nbl::hlsl::is_signed_v<T> && ::nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT floating_point = nbl::hlsl::is_floating_point_v<T> && nbl::hlsl::is_scalar_v<T>;

#endif

namespace impl
{
template<typename T>
struct IsEmulatingFloatingPointType
{
	static const bool value = false;
};
}

//! Floating point types are native floating point types or types that imitate native floating point types (for example emulated_float64_t)
template<typename T>
NBL_BOOL_CONCEPT FloatingPointLike = (nbl::hlsl::is_floating_point_v<T> && nbl::hlsl::is_scalar_v<T>) || impl::IsEmulatingFloatingPointType<T>::value;

}
}
}
#endif