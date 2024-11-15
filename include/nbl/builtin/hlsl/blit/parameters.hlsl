// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_


#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"


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
	float32_t referenceAlpha; //

	uint16_t3 inputDims; //
	uint16_t3 outputDims;
	uint16_t3 windowDims; //
	uint16_t3 phaseCount;
	uint16_t3 preloadRegion; //
	uint16_t3 iterationRegionXPrefixProducts;
	uint16_t3 iterationRegionYPrefixProducts;
	uint16_t3 iterationRegionZPrefixProducts;
};

// We do some dumb things with bitfields here like not using `vector<uint16_t,N>`, because AMD doesn't support them in push constants
struct SPerWorkgroup
{
	static inline SPerWorkgroup create(const float32_t3 _scale, const uint16_t _imageDim, const uint16_t3 output, const uint16_t3 preload, const uint16_t _secondScratchOffDWORD)
	{
		SPerWorkgroup retval;
		retval.scale = _scale;
		retval.imageDim = _imageDim;
		retval.preloadWidth = preload[0];
		retval.preloadHeight = preload[1];
		retval.outputWidth = output[0];
		retval.outputHeight = output[1];
		retval.outputDepth = output[2];
		retval.secondScratchOffDWORD = _secondScratchOffDWORD;
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

	inline uint16_t2 getPreloadExtentExceptLast() NBL_CONST_MEMBER_FUNC
	{
		return uint16_t2(preloadWidth,preloadHeight);
	}

	inline uint16_t getPhaseCount(const int32_t axis) NBL_CONST_MEMBER_FUNC
	{
		switch (axis)
		{
			case 2:
				return phaseCountZ;
				break;
			case 1:
				return phaseCountY;
				break;
			default:
				break;
		}
		return phaseCountX;
	}

	inline uint16_t getPassInvocationCount(const int32_t axis) NBL_CONST_MEMBER_FUNC
	{
		switch (axis)
		{
			case 2:
				return zPassInvocations;
				break;
			case 1:
				return yPassInvocations;
				break;
			default:
				break;
		}
		return xPassInvocations;
	}

	inline bool doCoverage() NBL_CONST_MEMBER_FUNC
	{
		return bool(coverageDWORD);
	}

	inline uint32_t3 getInputEndCoord() NBL_CONST_MEMBER_FUNC
	{
		return uint32_t3(inputEndX,inputEndX,inputEndZ);
	}

#ifndef __HLSL_VERSION
	explicit inline operator bool() const
	{
		return outputWidth && outputHeight && outputDepth && preloadWidth && preloadHeight && preloadLast;
	}
#endif

	// ratio of input pixels to output
	float32_t3 scale;
	// TODO: rename
	float32_t3 negativeSupport;
	// 16bit in each dimension because some GPUs actually have enough shared memory for 32k pixels per dimension
	// TODO: rename `output` to `perWGOutput`
	uint32_t outputWidth	: 16;
	uint32_t outputHeight	: 16;
	uint32_t outputDepth	: 16;
	// 16bit because we can theoretically have a very thin preload region
	uint32_t preloadWidth	: 16;
	uint32_t preloadHeight	: 16;
	// 64kb of smem is absolute max you'll see in the wild
	uint32_t preloadCount	: 16;
	// worst case is a phase of 2^16-1
	// while the last pass invocations need to be less than 64k because of the smem constraint
	uint32_t phaseCountX		: 16;
	uint32_t xPassInvocations	: 16;
	uint32_t phaseCountY		: 16;
	uint32_t yPassInvocations	: 16;
	uint32_t phaseCountZ		: 16;
	uint32_t zPassInvocations	: 16;
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint32_t secondScratchOffDWORD	: 14;
	//! coverage settings
	uint32_t inputEndX				: 17;
	uint32_t unused0				: 1;
	uint32_t inputEndY	: 17;
	uint32_t unused1	: 15;
	uint32_t inputEndZ		: 17;
	// whether its an image1D, image2D or image3D
	uint32_t imageDim		: 2;
	// saving a bit
	uint32_t lastChannel	: 2;
	uint32_t inLevel		: 5;
	uint32_t unused2		: 6;

	//! coverage settings
	uint32_t coverageChannel	: 2;
	uint32_t coverageDWORD		: 14;
	float32_t alphaRefValue;
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
	float32_t3 inputImageExtentRcp;
	uint32_t inputDescIx : 20;
	uint32_t samplerDescIx : 12;
	//
	uint32_t outputDescIx : 20;
	uint32_t unused0 : 12;
	//! coverage settings
	uint32_t unused1 : 12;
	uint32_t intermAlphaDescIx : 20;
	// required to compare the atomic count of passing pixels against, so we can get original coverage
	uint32_t inPixelCount;
};


}
}
}

#endif