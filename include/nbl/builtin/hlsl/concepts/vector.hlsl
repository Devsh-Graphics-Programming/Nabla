// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_VECTOR_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_VECTOR_HLSL_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

//! Concept for native vectors.
template<typename T>
NBL_BOOL_CONCEPT Vector = is_vector<T>::value;
template<typename T>
NBL_BOOL_CONCEPT FloatingPointVector = concepts::Vector<T> && concepts::FloatingPointScalar<typename vector_traits<T>::scalar_type>;
template<typename T>
NBL_BOOL_CONCEPT IntVector = concepts::Vector<T> && (is_integral_v<typename vector_traits<T>::scalar_type>);
template<typename T>
NBL_BOOL_CONCEPT SignedIntVector = concepts::Vector<T> && concepts::SignedIntegralScalar<typename vector_traits<T>::scalar_type>;

//! Concept for native vectors and vector like structs.
template<typename T>
NBL_BOOL_CONCEPT Vectorial = vector_traits<T>::IsVector;

#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T>
NBL_BOOL_CONCEPT FloatingPointVectorial = concepts::Vectorial<T> && concepts::FloatingPointScalar<typename vector_traits<T>::scalar_type>;
template<typename T>
NBL_BOOL_CONCEPT FloatingPointLikeVectorial = concepts::Vectorial<T> && concepts::FloatingPointLikeScalar<typename vector_traits<T>::scalar_type>;
template<typename T>
NBL_BOOL_CONCEPT IntVectorial = concepts::Vectorial<T> && (is_integral_v<typename vector_traits<T>::scalar_type>);
template<typename T>
NBL_BOOL_CONCEPT SignedIntVectorial = concepts::Vectorial<T> && concepts::SignedIntegralScalar<typename vector_traits<T>::scalar_type>;

}
}
}
#endif