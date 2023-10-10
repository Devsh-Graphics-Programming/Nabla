// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_NBL_HLSL_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_
#define _NBL_NBL_HLSL_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ndarray_addressing
{

uint32_t snakeCurve(NBL_CONST_REF_ARG(uint32_t3) coordinate, NBL_CONST_REF_ARG(uint32_t3) extents)
{
	return (coordinate.z * extents.y + coordinate.y) * extents.x + coordinate.x;
}

uint32_t snakeCurveInverse(uint32_t linearIndex, NBL_CONST_REF_ARG(uint32_t2) gridDimPrefixProduct)
{
	uint32_t3 index3D;

	index3D.z = linearIndex / gridDimPrefixProduct.y;

	const uint32_t tmp = linearIndex - (index3D.z * gridDimPrefixProduct.y);
	index3D.y = tmp / gridDimPrefixProduct.x;
	index3D.x = tmp - (index3D.y * gridDimPrefixProduct.x);

	return index3D;
}

uint32_t snakeCurveInverse(uint32_t linearIndex, NBL_CONST_REF_ARG(uint32_t3) gridDim)
{
	return snakeCurveInverse(linearIndex, uint32_t2(gridDim.x, gridDim.x*gridDim.y));
}

}
}
}
#endif