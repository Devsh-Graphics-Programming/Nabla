// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_ALPHA_TEST_INCLUDED_


#include <nbl/builtin/hlsl/blit/parameters.hlsl>
#include <nbl/builtin/hlsl/blit/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{

template <uint32_t BlitDimCount, typename StatisticsBuffer, typename InTexture>
inline void alpha_test(
	NBL_REF_ARG(StatisticsBuffer) statistics,
	NBL_CONST_REF_ARG(InTexture) inTexture,
	NBL_CONST_REF_ARG(parameters_t) params,
	NBL_CONST_REF_ARG(uint32_t3) dispatchThreadID,
	NBL_CONST_REF_ARG(uint32_t3) groupID)
{
	const uint32_t3 inDim = params.getInputImageDimensions();

	if (all(dispatchThreadID < inDim))
	{
		const float alpha = getData<BlitDimCount>(inTexture, dispatchThreadID, groupID.z).a;
		if (alpha > params.referenceAlpha)
			InterlockedAdd(statistics[groupID.z].passedPixelCount, uint32_t(1));
	}
}

}
}
}

#endif

