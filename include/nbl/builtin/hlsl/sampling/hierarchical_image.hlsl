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
struct LuminanceMapSampler
{
	using scalar_type = ScalarT;
	using vector2_type = vector<scalar_type, 2>;
	using vector4_type = vector<scalar_type, 4>;

	LuminanceAccessorT _map;
	uint32_t2 _mapSize;
  uint32_t2 _lastWarpPixel;
	bool _aspect2x1;

	static LuminanceMapSampler<ScalarT, LuminanceAccessorT> create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, uint32_t2 mapSize, bool aspect2x1, uint32_t2 warpSize)
	{
	  LuminanceMapSampler<ScalarT, LuminanceAccessorT> result;
	  result._map = lumaMap;
	  result._mapSize = mapSize;
    result._lastWarpPixel = warpSize - uint32_t2(1, 1);
	  result._aspect2x1 = aspect2x1;
	  return result;
	}

	static bool choseSecond(scalar_type first, scalar_type second, NBL_REF_ARG(scalar_type) xi)
	{
		// numerical resilience against IEEE754
		scalar_type dummy = scalar_type(0);
		PartitionRandVariable<scalar_type> partition;
		partition.leftProb = scalar_type(1) / (scalar_type(1) + (second / first));
		return partition(xi, dummy);
	}

	vector2_type binarySearch(const uint32_t2 coord)
	{
		// We use _lastWarpPixel here for corner sampling
    float32_t2 xi = float32_t2(coord)/ _lastWarpPixel;
		uint32_t2 p = uint32_t2(0, 0);
		const uint32_t2 mip2x1 = findMSB(_mapSize.y);

		if (_aspect2x1) {
			// do one split in the X axis first cause penultimate full mip would have been 2x1
			p.x = choseSecond(_map.get(uint32_t2(0, 0), mip2x1), _map.get(uint32_t2(1, 0), mip2x1), xi.x) ? 1 : 0;
		}

		for (int i = mip2x1 - 1; i >= 0; i--)
		{
			p <<= 1;
			const vector4_type values = _map.gather(p, i);
			scalar_type wx_0, wx_1;
			{
				const scalar_type wy_0 = values[3] + values[2];
				const scalar_type wy_1 = values[1] + values[0];
				if (choseSecond(wy_0, wy_1, xi.y))
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
      if (choseSecond(wx_0, wx_1, xi.x))
        p.x |= 1;
		}


		// If we don`t add xi, the sample will clump to the lowest corner of environment map texel. We add xi to simulate uniform distribution within a pixel and make the sample continuous. This is why we compute the pdf not from the normalized luminance of the texel, instead from the reciprocal of the Jacobian.
		const vector2_type directionUV = (vector2_type(p.x, p.y) + xi) / vector2_type(_mapSize);
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
struct HierarchicalImage 
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

	static HierarchicalImage create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, NBL_CONST_REF_ARG(HierarchicalSamplerT) warpMap, uint32_t2 warpSize, scalar_type avgLuma) 
	{
		HierarchicalImage<ScalarT, LuminanceAccessorT, HierarchicalSamplerT, PostWarpT> result;
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
