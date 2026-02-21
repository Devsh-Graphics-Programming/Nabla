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
	float32_t3 fScale; //
	float32_t3 negativeSupport;
	float32_t referenceAlpha;
	uint32_t kernelWeightsOffsetY;
	uint32_t kernelWeightsOffsetZ;
	uint32_t inPixelCount;
	uint32_t outPixelCount;

	uint16_t3 inputDims;
	uint16_t3 outputDims;
	uint16_t3 windowDims; //
	uint16_t3 phaseCount;
	uint16_t3 preloadRegion; //
	uint16_t3 iterationRegionXPrefixProducts;
	uint16_t3 iterationRegionYPrefixProducts;
	uint16_t3 iterationRegionZPrefixProducts;

	uint16_t secondScratchOffset; //
	uint16_t outputTexelsPerWGZ; //

	uint32_t3 getOutputTexelsPerWG()
	{
		//! `outputTexelsPerWG.xy` just happens to be in the first components of `iterationRegionsXPrefixProducts` and `iterationRegionYPrefixProducts` --this is
		//! the result of how we choose to iterate, i.e. if, in the future, we decide to iterate differently, this needs to change.
		return uint32_t3(iterationRegionXPrefixProducts.x, iterationRegionYPrefixProducts.x, outputTexelsPerWGZ);
	}
};

// We do some dumb things with bitfields here like not using `vector<uint16_t,N>`, because AMD doesn't support them in push constants
struct SPerWorkgroup
{
	static inline SPerWorkgroup create(const float32_t3 _scale, const uint16_t _imageDim, const uint16_t3 output, const uint16_t3 preload, const uint16_t _otherPreloadOffset)
	{
		SPerWorkgroup retval;
		retval.scale = _scale;
		retval.imageDim = _imageDim;
		retval.preloadWidth = preload[0];
		retval.preloadHeight = preload[1];
		retval.preloadDepth = preload[2];
		retval.outputWidth = output[0];
		retval.outputHeight = output[1];
		retval.outputDepth = output[2];
		retval.otherPreloadOffset = _otherPreloadOffset;
		return retval;
	}

	inline uint16_t3 getOutputBaseCoord(const uint16_t3 workgroup) NBL_CONST_MEMBER_FUNC
	{
		return workgroup*uint16_t3(outputWidth,outputHeight,outputDepth);
	}

	inline uint16_t3 getWorkgroupCount(const uint16_t3 outExtent, const uint16_t layersToBlit=0) NBL_CONST_MEMBER_FUNC
	{
		const uint16_t3 unit = uint16_t3(1,1,1);
		uint16_t3 retval = unit;
		retval += (outExtent-unit)/getOutputBaseCoord(unit);
		if (layersToBlit)
			retval[2] = layersToBlit;
		return retval;
	}

#ifndef __HLSL_VERSION
	explicit inline operator bool() const
	{
		return outputWidth && outputHeight && outputDepth && preloadWidth && preloadHeight && preloadDepth;
	}
#endif

	// ratio of input pixels to output
	float32_t3 scale;
	// whether its an image1D, image2D or image3D
	uint32_t imageDim : 2;
	uint32_t unused0 : 14; // channel, iterationRegionPrefixSums ?
	// 16bit in each dimension because some GPUs actually have enough shared memory for 32k pixels
	uint32_t outputWidth	: 16;
	uint32_t outputHeight	: 16;
	uint32_t outputDepth	: 16;
	uint32_t preloadWidth		: 16;
	uint32_t preloadHeight		: 16;
	uint32_t preloadDepth		: 16;
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint32_t otherPreloadOffset	: 16;
};

struct Parameters
{
#ifndef __HLSL_VERSION
	explicit inline operator bool() const
	{
		return bool(perWG);
	}
#endif

	SPerWorkgroup perWG; // rename to perBlitWG? 
	//! general settings
	uint32_t inputDescIx : 19;
	uint32_t samplerDescIx : 11;
	uint32_t unused0 : 2;
	//
	uint32_t outputDescIx : 19;
	uint32_t channelCount : 3;
	uint32_t unused1 : 10;
	//
	uint32_t unused2 : 12;
	//! coverage settings
	uint32_t intermAlphaDescIx : 19;
	uint32_t coverage : 1;
	// required to compare the atomic count of passing pixels against, so we can get original coverage
	uint32_t inPixelCount;
};


}
}
}

#endif