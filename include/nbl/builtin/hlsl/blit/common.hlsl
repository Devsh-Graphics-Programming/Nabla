// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{
namespace impl
{

template <uint32_t Dimension>
struct dim_to_image_properties { };

template <>
struct dim_to_image_properties<1>
{
	using combined_sampler_t = Texture1DArray<float4>;
	using image_t = RWTexture1DArray<float4>;

	template <typename T>
	static vector<T, 2> getIndexCoord(vector<T, 3> coords, uint32_t layer)
	{
		return vector<T, 2>(coords.x, layer);
	}
};

template <>
struct dim_to_image_properties<2>
{
	using combined_sampler_t = Texture2DArray<float4>;
	using image_t = RWTexture2DArray<float4>;

	template <typename T>
	static vector<T,3> getIndexCoord(vector<T, 3> coords, uint32_t layer)
	{
		return vector<T, 3>(coords.xy, layer);
	}
};

template <>
struct dim_to_image_properties<3>
{
	using combined_sampler_t = Texture3D<float4>;
	using image_t = RWTexture3D<float4>;

	template <typename T>
	static vector<T, 3> getIndexCoord(vector<T, 3> coords, uint32_t layer)
	{
		return vector<T,3>(coords);
	}
};

}


template<
	uint32_t _WorkGroupSizeX,
	uint32_t _WorkGroupSizeY,
	uint32_t _WorkGroupSizeZ,
	uint32_t _SMemFloatsPerChannel,
	uint32_t _BlitOutChannelCount,
	uint32_t _BlitDimCount,
	uint32_t _AlphaBinCount>
struct consteval_parameters_t
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SMemFloatsPerChannel = _SMemFloatsPerChannel;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t BlitOutChannelCount = _BlitOutChannelCount;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t BlitDimCount = _BlitDimCount;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t AlphaBinCount = _AlphaBinCount;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkGroupSizeX = _WorkGroupSizeX;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkGroupSizeY = _WorkGroupSizeY;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkGroupSizeZ = _WorkGroupSizeZ;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkGroupSize = WorkGroupSizeX * WorkGroupSizeY * WorkGroupSizeZ;
};

}
}
}

#endif