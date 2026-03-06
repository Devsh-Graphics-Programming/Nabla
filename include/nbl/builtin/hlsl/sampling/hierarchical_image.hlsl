// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_

#include <nbl/builtin/hlsl/sampling/basic.hlsl>
#include <nbl/builtin/hlsl/sampling/warp.hlsl>
#include <nbl/builtin/hlsl/sampling/hierarchical_image/accessors.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template <typename ScalarT, typename LuminanceAccessorT 
  NBL_PRIMARY_REQUIRES(
		is_scalar_v<ScalarT> && 
		hierarchical_image::LuminanceReadAccessor<LuminanceAccessorT, ScalarT>
	)
struct HierarchicalLuminanceSampler
{
	using scalar_type = ScalarT;
	using vector2_type = vector<scalar_type, 2>;
	using vector4_type = vector<scalar_type, 4>;

	LuminanceAccessorT _map;
	float32_t2 _rcpWarpSize;
	uint16_t2 _mapSize;
	uint16_t _mip2x1 : 15;
	uint16_t _aspect2x1 : 1;

	static HierarchicalLuminanceSampler<ScalarT, LuminanceAccessorT> create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, uint32_t2 mapSize, bool aspect2x1, uint32_t2 warpSize)
	{
	  HierarchicalLuminanceSampler<ScalarT, LuminanceAccessorT> result;
	  result._map = lumaMap;
		result._mapSize = vector2_type(mapSize);
		result._rcpWarpSize = scalar_type(1.0) / vector2_type(warpSize - uint32_t2(1, 1));
		// Note: We use mapSize.y here because the currently the map aspect ratio can only be 1x1 or 2x1
		result._mip2x1 = findMSB(mapSize.y);
		result._aspect2x1 = aspect2x1;
	  return result;
	}

	static bool __choseSecond(scalar_type first, scalar_type second, NBL_REF_ARG(scalar_type) xi)
	{
		// numerical resilience against IEEE754
		scalar_type dummy = scalar_type(0);
		PartitionRandVariable<scalar_type> partition;
		partition.leftProb = scalar_type(1) / (scalar_type(1) + (second / first));
		return partition(xi, dummy);
	}

	vector2_type binarySearch(const uint32_t2 coord)
	{
		// We use _rcpWarpSize here for corner sampling. Corner sampling is a sampling mechanism where we map texel_index / map_size to the center of the texel instead of the edge of the texel. So uv.x == 0 is mapped to the center of the left most texel, and uv.x == width - 1 is mapped to the center of the right most texel. That's why the length of the domain is subtracted by 1 for each dimension.
    float32_t2 xi = float32_t2(coord) * _rcpWarpSize;
		uint32_t2 p = uint32_t2(0, 0);

		if (_aspect2x1) {
			// do one split in the X axis first cause penultimate full mip would have been 2x1
			p.x = __choseSecond(_map.texelFetch(uint32_t2(0, 0), _mip2x1), _map.texelFetch(uint32_t2(1, 0), _mip2x1), xi.x) ? 1 : 0;
		}

		for (int i = _mip2x1 - 1; i >= 0; i--)
		{
			p <<= 1;
			const vector4_type values = _map.texelGather(p, i);
			scalar_type wx_0, wx_1;
			{
				const scalar_type wy_0 = values[3] + values[2];
				const scalar_type wy_1 = values[1] + values[0];
				if (__choseSecond(wy_0, wy_1, xi.y))
				{
					p.y |= 1;
					wx_0 = values[0];
					wx_1 = values[1];
				}
				else
				{
					wx_0 = values[3];
					wx_1 = values[2];
				}
      }
      if (__choseSecond(wx_0, wx_1, xi.x))
        p.x |= 1;
		}


		// If we don`t add xi, the sample will clump to the lowest corner of environment map texel. Each time we call PartitionRandVariable(), the output xi is the new xi that determines how left and right(or top and bottom for y axis) to choose the child partition. It means that if for some input xi, the output xi = 0, then the input xi is the edge of choosing this partition and the previous partition, and vice versa, if output xi = 1, then the input xi is the edge of choosing this partition and the next partition. Hence, by adding xi to the lower corner of the texel, we create a gradual transition from one pixel to another. Without adding output xi, the calculation of jacobian using the difference of sample value would not work.
		// Since we want to do corner sampling. We have to handle edge texels as corner cases. Remember, in corner sampling we map uv [0,1] to [center of first texel, center of last texel]. So when p is an edge texel, we have to remap xi. [0.5, 1] when p == 0, and [0.5, 1] when p == length - 1.
		if (p.x == 0)
			xi.x = xi.x * scalar_type(0.5) + scalar_type(0.5);
		if (p.y == 0)
			xi.y = xi.y * scalar_type(0.5) + scalar_type(0.5);
		if (p.x == _mapSize.x - 1)
			xi.x = xi.x * scalar_type(0.5);
		if (p.y == _mapSize.y - 1)
			xi.y = xi.y * scalar_type(0.5);

		const vector2_type directionUV = (vector2_type(p.x, p.y) + xi) / _mapSize;
		return directionUV;
	}

	matrix<scalar_type, 4, 2> sampleUvs(uint32_t2 sampleCoord) NBL_CONST_MEMBER_FUNC
	{
		const vector2_type dir0 = binarySearch(sampleCoord + vector2_type(0, 1));
		const vector2_type dir1 = binarySearch(sampleCoord + vector2_type(1, 1));
		const vector2_type dir2 = binarySearch(sampleCoord + vector2_type(1, 0));
		const vector2_type dir3 = binarySearch(sampleCoord);
		return matrix<scalar_type, 4, 2>(
			dir0,
			dir1,
			dir2,
			dir3
		);
	}
};

template <typename ScalarT, typename LuminanceAccessorT, typename HierarchicalSamplerT, typename PostWarpT 
  NBL_PRIMARY_REQUIRES(is_scalar_v<ScalarT> &&
		concepts::accessors::GenericReadAccessor<LuminanceAccessorT, ScalarT, float32_t2> &&
		hierarchical_image::HierarchicalSampler<HierarchicalSamplerT, ScalarT> &&
		concepts::Warp<PostWarpT>)
struct WarpmapSampler 
{
	using scalar_type = ScalarT;
	using vector2_type = vector<ScalarT, 2>;
	using vector3_type = vector<ScalarT, 3>;
	using vector4_type = vector<ScalarT, 4>;
	LuminanceAccessorT _lumaMap;
	HierarchicalSamplerT _warpMap;
	uint32_t2 _warpSize;
	uint32_t2 _lastWarpPixel;
	scalar_type _rcpAvgLuma;

	static WarpmapSampler create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, NBL_CONST_REF_ARG(HierarchicalSamplerT) warpMap, uint32_t2 warpSize, scalar_type avgLuma) 
	{
		WarpmapSampler<ScalarT, LuminanceAccessorT, HierarchicalSamplerT, PostWarpT> result;
		result._lumaMap = lumaMap;
		result._warpMap = warpMap;
		result._warpSize = warpSize;
		result._lastWarpPixel = warpSize - uint32_t2(1, 1);
		result._rcpAvgLuma = ScalarT(1.0) / avgLuma;
		return result;
	}

	vector2_type inverseWarp_and_deferredPdf(NBL_REF_ARG(scalar_type) pdf, vector3_type direction) NBL_CONST_MEMBER_FUNC
  {
		vector2_type envmapUv = PostWarpT::inverseWarp(direction);
		scalar_type luma;
		_lumaMap.get(envmapUv, luma);
		pdf = (luma * _rcpAvgLuma) * PostWarpT::backwardDensity(direction);
		return envmapUv;
  }

	scalar_type deferredPdf(vector3_type direction) NBL_CONST_MEMBER_FUNC
	{
		vector2_type envmapUv = PostWarpT::inverseWarp(direction);
		scalar_type luma;
		_lumaMap.get(envmapUv, luma);
		return luma * _rcpAvgLuma * PostWarpT::backwardDensity(direction);
	}

	vector3_type generate_and_pdf(NBL_REF_ARG(scalar_type) pdf, NBL_REF_ARG(vector2_type) uv, vector2_type xi) NBL_CONST_MEMBER_FUNC
	{
		const vector2_type texelCoord = xi * float32_t2(_lastWarpPixel);

		matrix<scalar_type, 4, 2> uvs = _warpMap.sampleUvs(uint32_t2(texelCoord));

		const vector2_type interpolant = frac(texelCoord);

		const vector2_type xDiffs[] = {
			uvs[2] - uvs[3],
			uvs[1] - uvs[0]
		};
		const vector2_type yVals[] = {
			xDiffs[0] * interpolant.x + uvs[3],
			xDiffs[1] * interpolant.x + uvs[0]
		};
		const vector2_type yDiff = yVals[1] - yVals[0];
		uv = yDiff * interpolant.y + yVals[0];

		const WarpResult<vector3_type> warpResult = PostWarpT::warp(uv);

		const scalar_type detInterpolJacobian = determinant(matrix<scalar_type, 2, 2>(
			lerp(xDiffs[0], xDiffs[1], interpolant.y), // first column dFdx
			yDiff // second column dFdy
		)) * _lastWarpPixel.x * _lastWarpPixel.y;

		pdf = abs(warpResult.density / detInterpolJacobian);

		return warpResult.dst;
	}
};

}
}
}

#endif
