// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace blit
{

struct parameters_t
{

	float32_t3 fScale;
	float32_t3 negativeSupport;
	float32_t referenceAlpha;
	uint32_t kernelWeightsOffsetY;
	uint32_t kernelWeightsOffsetZ;
	uint32_t inPixelCount;
	uint32_t outPixelCount;

	uint16_t3 inputDims;
	uint16_t3 outputDims;
	uint16_t3 windowDims;
	uint16_t3 phaseCount;
	uint16_t3 preloadRegion;
	uint16_t3 iterationRegionXPrefixProducts;
	uint16_t3 iterationRegionYPrefixProducts;
	uint16_t3 iterationRegionZPrefixProducts;

	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint16_t secondScratchOffset;
	uint16_t outputTexelsPerWGZ;

	uint32_t3 getOutputTexelsPerWG()
	{
		//! `outputTexelsPerWG.xy` just happens to be in the first components of `iterationRegionsXPrefixProducts` and `iterationRegionYPrefixProducts` --this is
		//! the result of how we choose to iterate, i.e. if, in the future, we decide to iterate differently, this needs to change.
		return uint32_t3(iterationRegionXPrefixProducts.x, iterationRegionYPrefixProducts.x, outputTexelsPerWGZ);
	}
};


}
}
}

#endif