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
	uint32_t3 dims; // input dimensions in lower 16 bits, output dimensions in higher 16 bits
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint32_t secondScratchOffset;
	uint32_t3 iterationRegionXPrefixProducts;
	float referenceAlpha;
	float3 fScale;
	uint32_t inPixelCount;
	float3 negativeSupport;
	uint32_t outPixelCount;
	uint32_t3 windowDimPhaseCount; // windowDim in lower 16 bits, phaseCount in higher 16 bits
	uint32_t kernelWeightsOffsetY;
	uint32_t3 iterationRegionYPrefixProducts;
	uint32_t kernelWeightsOffsetZ;
	uint32_t3 iterationRegionZPrefixProducts;
	uint32_t outputTexelsPerWGZ;
	uint32_t3 preloadRegion;

	uint32_t3 getInputImageDimensions()
	{
		NBL_CONSTEXPR uint32_t mask = ((1u << 16) - 1);
		return dims & mask;
	}

	uint32_t3 getOutputImageDimensions()
	{
		return dims >> 16;
	}

	uint32_t3 getWindowDimensions()
	{
		NBL_CONSTEXPR uint32_t mask = ((1u << 16) - 1);
		return windowDimPhaseCount & mask;
	}

	uint32_t3 getPhaseCount()
	{
		return windowDimPhaseCount >> 16;
	}

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