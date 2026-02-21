// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_MATRIX_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_MATRIX_HLSL_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

template<typename T>
NBL_BOOL_CONCEPT Matrix = is_matrix<T>::value;

template<typename T>
NBL_BOOL_CONCEPT Matricial = matrix_traits<T>::IsMatrix;

}
}
}
#endif