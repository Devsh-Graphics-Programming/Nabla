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

template<typename T, typename U>
NBL_BOOL_CONCEPT same_as = is_same_v<T, U>;

template<typename T>
NBL_BOOL_CONCEPT Integral = nbl::hlsl::is_integral_v<T>;

template<typename T>
NBL_BOOL_CONCEPT SignedIntegral = nbl::hlsl::is_signed_v<T> && nbl::hlsl::is_integral_v<T>;

template<typename T>
NBL_BOOL_CONCEPT UnsignedIntegral = !nbl::hlsl::is_signed_v<T> && ::nbl::hlsl::is_integral_v<T>;

template<typename T>
NBL_BOOL_CONCEPT FloatingPoint = nbl::hlsl::is_floating_point_v<T>;

template<typename T>
NBL_BOOL_CONCEPT Boolean = nbl::hlsl::is_same_v<T, bool> || (nbl::hlsl::is_vector_v<T> && nbl::hlsl::is_same_v<typename vector_traits<T>::scalar_type, bool>);

template <typename T>
NBL_BOOL_CONCEPT Scalar = nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT IntegralScalar = nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT SignedIntegralScalar = nbl::hlsl::is_signed_v<T> && nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT UnsignedIntegralScalar = !nbl::hlsl::is_signed_v<T> && ::nbl::hlsl::is_integral_v<T> && nbl::hlsl::is_scalar_v<T>;

template<typename T>
NBL_BOOL_CONCEPT FloatingPointScalar = nbl::hlsl::is_floating_point_v<T> && nbl::hlsl::is_scalar_v<T>;

// TODO: implement when hlsl::is_base_of is done
//#define NBL_CONCEPT_NAME DerivedFrom
// ...

// TODO: implement when hlsl::is_converible is done
//#define NBL_CONCEPT_NAME ConvertibleTo
// ...

// TODO?
//#define NBL_CONCEPT_NAME AssignableFrom

// TODO?
//template <typename T, typename U>
//concept common_with = std::common_with<T, U>;

namespace impl
{
template<typename T>
struct IsEmulatingFloatingPointType
{
	static const bool value = nbl::hlsl::is_floating_point_v<T>;
};
}

//! Floating point types are native floating point types or types that imitate native floating point types (for example emulated_float64_t)
template<typename T>
NBL_BOOL_CONCEPT FloatingPointLikeScalar = impl::IsEmulatingFloatingPointType<T>::value;

}
}
}
#endif