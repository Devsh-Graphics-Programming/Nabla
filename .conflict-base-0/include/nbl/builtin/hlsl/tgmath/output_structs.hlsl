// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_SPIRV_INTRINSICS_OUTPUT_STRUCTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_SPIRV_INTRINSICS_OUTPUT_STRUCTS_INCLUDED_

#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct ModfOutput;

template<typename T>  NBL_PARTIAL_REQ_TOP(concepts::FloatingPoint<T>)
struct ModfOutput<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPoint<T>) >
{
	T fractionalPart;
	T wholeNumberPart;
};

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct FrexpOutput;

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct FrexpOutput<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
	T significand;
	int exponent;
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointVector<T>)
struct FrexpOutput<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointVector<T>) >
{
	T significand;
	vector<int, vector_traits<T>::Dimension> exponent;
};

}
}

#endif