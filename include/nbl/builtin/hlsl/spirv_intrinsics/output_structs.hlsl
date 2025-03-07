// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_OUTPUT_STRUCTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_OUTPUT_STRUCTS_INCLUDED_

#include <nbl/builtin/hlsl/concepts/core.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct AddCarryOutput;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct SubBorrowOutput;


template<typename T>  NBL_PARTIAL_REQ_TOP(concepts::UnsignedIntegral<T>)
struct AddCarryOutput<T NBL_PARTIAL_REQ_BOT(concepts::UnsignedIntegral<T>) >
{
	T result;
	T carry;
};

template<typename T>  NBL_PARTIAL_REQ_TOP(concepts::UnsignedIntegral<T>)
struct SubBorrowOutput<T NBL_PARTIAL_REQ_BOT(concepts::UnsignedIntegral<T>) >
{
	T result;
	T borrow;
};
}
}

#endif