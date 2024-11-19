// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_PARAMETERS_INCLUDED_


#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/ndarray_addressing.hlsl"


namespace nbl
{
namespace hlsl
{
namespace blit
{

// We do some dumb things with bitfields here like not using `vector<uint16_t,N>`, because AMD doesn't support them in push constants
struct SPerWorkgroup
{
	//
	template<int32_t Dims>
	struct PatchLayout
	{
		inline uint16_t getLinearEnd() NBL_CONST_MEMBER_FUNC
		{
			return value[2];
		}

		inline uint16_t getIndex(const vector<uint16_t,Dims> id) NBL_CONST_MEMBER_FUNC
		{
			return ndarray_addressing::snakeCurve(id,value);
		}

		inline vector<uint16_t,Dims> getID(const uint16_t linearIndex) NBL_CONST_MEMBER_FUNC
		{
			return ndarray_addressing::snakeCurveInverse<Dims,uint16_t,uint16_t>(linearIndex,value);
		}

		vector<uint16_t,Dims> value;
	};

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

	template<int32_t Dims=3>
	vector<uint16_t,Dims> getPerWGOutputExtent() NBL_CONST_MEMBER_FUNC;

	inline uint16_t3 getWorkgroupCount(const uint16_t3 outExtent, const uint16_t layersToBlit=0) NBL_CONST_MEMBER_FUNC
	{
		const uint16_t3 unit = uint16_t3(1,1,1);
		uint16_t3 retval = unit;
		retval += (outExtent-unit)/getPerWGOutputExtent();
		if (layersToBlit)
			retval[2] = layersToBlit;
		return retval;
	}

	// tells you the first pixel in the other image that's either the same or with a greater coordinate when snapped to the other grid
	inline float32_t inputUpperBound(const uint16_t coord, const int32_t axis)
	{
		return ceil(float32_t(coord)*scale[axis]+halfScalePlusMinSupportMinusHalf[axis]);
	}
	template<int32_t N>
	inline vector<float32_t,N> inputUpperBound(const vector<uint16_t,N> coord)
	{
		return ceil(vector<float32_t,N>(coord)*scale+truncate<N>(halfScalePlusMinSupportMinusHalf));
	}

	//
	template<int32_t Dims>
	inline PatchLayout<Dims> getPreloadMeta() NBL_CONST_MEMBER_FUNC
	{
		PatchLayout<Dims> retval;
		retval.value[Dims-1] = preloadDepth;
		if (Dims>1)
		{
			retval.value[0] = preloadWidth;
			retval.value[Dims-1] *= retval.value[0];
			// if any problems with indexing OOB, then explicitly specialize
			if (Dims>2)
			{
				retval.value[1] = preloadHeight;
				retval.value[Dims-1] *= retval.value[1];
			}
		}
		return retval;
	}
	
	template<int32_t Dims=3>
	vector<uint16_t,Dims> getWindowExtent() NBL_CONST_MEMBER_FUNC;

	// We sweep along X, then Y, then Z, at every step we need the loads from smem to be performed on consecutive values so that we don't have bank conflicts
	// 1st pass input: we output oW x pH x pD, X must be minor on input, for cheap `snakeCurveInverse` for 1D and 2D cases, do XYZ order
	// 2nd pass input: we output oW x oH x pD, Y must be minor on input, for cheap `snakeCurveInverse` for 2D case, do YXZ order
	// 3rd pass input: we output oW x oH x oD, Z must be minor on input, order can be ZYX or ZXY, but settled on ZYX
	template<int32_t Dims>
	PatchLayout<Dims> getPassMeta(const int32_t axis) NBL_CONST_MEMBER_FUNC;

	// can only be called with `axis<Dims-1`
	template<int32_t Dims>
	uint16_t getStorageIndex(const int32_t axis, const vector<uint16_t,Dims> coord) NBL_CONST_MEMBER_FUNC;

	template<int32_t Dims>
	static vector<uint16_t,Dims> unswizzle(const vector<uint16_t,Dims> coord);

	inline uint16_t getPhaseCount(const int32_t axis) NBL_CONST_MEMBER_FUNC
	{
		return axis!=0 ? (axis!=1 ? phaseCountZ:phaseCountY):phaseCountX;
	}

	inline bool doCoverage() NBL_CONST_MEMBER_FUNC
	{
		return bool(coverageDWORD);
	}

	template<int32_t Dims=3>
	vector<uint16_t,Dims> getInputMaxCoord() NBL_CONST_MEMBER_FUNC;

#ifndef __HLSL_VERSION
	explicit inline operator bool() const
	{
		return outputWidth && outputHeight && outputDepth && preloadWidth && preloadHeight && preloadDepth;
	}
#endif

	// ratio of input pixels to output
	float32_t3 scale;
	// `0.5*scale+minSupport-0.5`
	float32_t3 halfScalePlusMinSupportMinusHalf;
	// 16bit in each dimension because some GPUs actually have enough shared memory for 32k pixels per dimension
	// TODO: rename `output` to `perWGOutput`
	uint32_t outputWidth	: 16;
	uint32_t outputHeight	: 16;
	uint32_t outputDepth	: 16;
	// 16bit because we can theoretically have a very thin preload region, but 64kb of smem is absolute max you'll see in the wild
	uint32_t preloadWidth	: 16;
	uint32_t preloadHeight	: 16;
	uint32_t preloadDepth	: 16;
	// kernel gather area for a single output texel
	uint32_t windowWidth	: 16;
	uint32_t windowHeight	: 16;
	uint32_t windowDepth	: 16;
	// worst case is a phase of 2^16-1
	uint32_t phaseCountX		: 16;
	uint32_t phaseCountY		: 16;
	uint32_t phaseCountZ		: 16;
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint32_t secondScratchOffDWORD	: 14;
	// whether its an image1D, image2D or image3D
	uint32_t imageDim				: 2;
	// saving a bit
	uint32_t lastChannel			: 2;
	uint32_t inLevel				: 5;
	uint32_t unused2				: 9;

	//! coverage settings
	uint32_t coverageDWORD		: 14;
	uint32_t coverageChannel	: 2;
	uint32_t inputMaxX			: 16;
	uint32_t inputMaxY : 16;
	uint32_t inputMaxZ : 16;
	float32_t alphaRefValue;
};

template<>
inline uint16_t1 SPerWorkgroup::getPerWGOutputExtent<1>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t1(outputWidth);
}
template<>
inline uint16_t2 SPerWorkgroup::getPerWGOutputExtent<2>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t2(outputWidth,outputHeight);
}
template<>
inline uint16_t3 SPerWorkgroup::getPerWGOutputExtent<3>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t3(outputWidth,outputHeight,outputDepth);
}

template<>
inline uint16_t1 SPerWorkgroup::getWindowExtent<1>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t1(windowWidth);
}
template<>
inline uint16_t2 SPerWorkgroup::getWindowExtent<2>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t2(windowWidth,windowHeight);
}
template<>
inline uint16_t3 SPerWorkgroup::getWindowExtent<3>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t3(windowWidth,windowHeight,windowDepth);
}

template<>
inline SPerWorkgroup::PatchLayout<1> SPerWorkgroup::getPassMeta<1>(const int32_t axis) NBL_CONST_MEMBER_FUNC
{
	PatchLayout<1> retval;
	retval.value = uint16_t1(outputWidth);
	return retval;
}
// TODO: eliminate the potential for bank conflicts during storage by making sure `outputHeight` used for snake curve addressing is odd 
template<>
inline SPerWorkgroup::PatchLayout<2> SPerWorkgroup::getPassMeta<2>(const int32_t axis) NBL_CONST_MEMBER_FUNC
{
	PatchLayout<2> retval;
	if (axis==0) // XY
	{
		retval.value[0] = outputWidth;
		retval.value[1] = preloadHeight;
	}
	else // YX
	{
		retval.value[0] = outputHeight;
		retval.value[1] = outputWidth;
	}
	retval.value[1] *= retval.value[0];
	return retval;
}
// TODO: eliminate the potential for bank conflicts during storage by making sure `outputHeight` and `outputDepth` used for snake curve addressing is odd 
template<>
inline SPerWorkgroup::PatchLayout<3> SPerWorkgroup::getPassMeta<3>(const int32_t axis) NBL_CONST_MEMBER_FUNC
{
	PatchLayout<3> retval;
	if (axis==0) // XYZ
	{
		retval.value[0] = outputWidth;
		retval.value[1] = preloadHeight;
		retval.value[2] = preloadDepth;
	}
	else
	{
		if (axis==1) // YXZ
		{
			retval.value[0] = outputHeight;
			retval.value[1] = outputWidth;
			retval.value[2] = preloadDepth;
		}
		else // ZYX or ZXY, ZYX may cause less bank conflicts if preaload and output extents are both PoT
		{
			retval.value[0] = outputDepth;
			retval.value[1] = outputHeight;
			retval.value[2] = outputWidth;
		}
	}
	retval.value[2] *= retval.value[1]*retval.value[0];
	return retval;
}

// have to specialize the Dims=1 case otherwise code won't compile
template<>
inline uint16_t SPerWorkgroup::getStorageIndex<1>(const int32_t axis, const uint16_t1 coord) NBL_CONST_MEMBER_FUNC
{
	return coord[0];
}
template<>
inline uint16_t SPerWorkgroup::getStorageIndex<2>(const int32_t axis, const uint16_t2 coord) NBL_CONST_MEMBER_FUNC
{
	return coord[0]*preloadHeight+coord[1];
}
template<>
inline uint16_t SPerWorkgroup::getStorageIndex<3>(const int32_t axis, const uint16_t3 coord) NBL_CONST_MEMBER_FUNC
{
	if (axis==0) // XYZ was the layout, prepping for YXZ
		return (coord[2]*outputWidth+coord[0])*preloadHeight+coord[1];
	// YXZ was the layout, prepping for ZYX
	return (coord[1]*outputHeight+coord[0])*preloadDepth+coord[2];
}

template<>
inline uint16_t1 SPerWorkgroup::unswizzle<1>(const uint16_t1 coord)
{
	return coord;
}
template<>
inline uint16_t2 SPerWorkgroup::unswizzle<2>(const uint16_t2 coord)
{
	return coord.yx; // YX -> XY
}
template<>
inline uint16_t3 SPerWorkgroup::unswizzle<3>(const uint16_t3 coord)
{
	return coord.zyx; // ZYX -> XYZ
}

template<>
inline uint16_t1 SPerWorkgroup::getInputMaxCoord<1>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t1(inputMaxX);
}
template<>
inline uint16_t2 SPerWorkgroup::getInputMaxCoord<2>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t2(inputMaxX,inputMaxY);
}
template<>
inline uint16_t3 SPerWorkgroup::getInputMaxCoord<3>() NBL_CONST_MEMBER_FUNC
{
	return uint16_t3(inputMaxX,inputMaxY,inputMaxZ);
}


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