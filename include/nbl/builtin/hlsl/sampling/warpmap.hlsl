// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_WARPMAP_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_WARPMAP_INCLUDED_

#include <nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/sampling/basic.hlsl>
#include <nbl/builtin/hlsl/sampling/concepts.hlsl>
#include <nbl/builtin/hlsl/sampling/hierarchical_image/accessors.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template <typename LuminanceAccessorT, typename HierarchicalSamplerT, typename PostWarpT 
  NBL_PRIMARY_REQUIRES(
    hierarchical_image::LuminanceReadAccessor<LuminanceAccessorT> &&
    hierarchical_image::WarpAccessor<HierarchicalSamplerT> &&
    concepts::BijectiveSampler<PostWarpT>)
struct WarpmapSampler 
{
  using scalar_type = typename LuminanceAccessorT::value_type;
  using vector2_type = vector<scalar_type, 2>;
  using vector3_type = vector<scalar_type, 3>;
  using vector4_type = vector<scalar_type, 4>;
  using domain_type = vector2_type;
  using codomain_type = vector3_type;
  using weight_type = scalar_type;
  using density_type = scalar_type;
  using this_type = WarpmapSampler<LuminanceAccessorT, HierarchicalSamplerT, PostWarpT>;
  struct cache_type
  {
    vector2_type xDiffs[2];
    vector2_type yDiff;
    vector2_type warpedUv;
    scalar_type interpolantY;
    typename PostWarpT::cache_type postWarpCache;
  };

  LuminanceAccessorT _lumaMap;
  HierarchicalSamplerT _warpMap;
  uint16_t2 _lastTexel;
  PostWarpT _postWarp;

  static WarpmapSampler create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, NBL_CONST_REF_ARG(HierarchicalSamplerT) warpMap) 
  {
    this_type result;
    result._lumaMap = lumaMap;
    result._warpMap = warpMap;
    result._lastTexel = warpMap.resolution() - uint16_t2(1, 1);
    return result;
  }


  codomain_type generate(const domain_type xi, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
  {
    vector2_type texelCoord = xi * vector2_type(_lastTexel.x, _lastTexel.y);
    vector2_type interpolant = hlsl::fract(texelCoord);
    float32_t2 warpmapUv = (texelCoord + float32_t2(0.5, 0.5)) / vector2_type(_warpMap.resolution());

    matrix<typename HierarchicalSamplerT::scalar_type, 4, 2> uvs;
    _warpMap.gatherUv(warpmapUv, uvs);

    const vector2_type xDiffs[] = {
      uvs[2] - uvs[3],
      uvs[1] - uvs[0]
    };
    const vector2_type yVals[] = {
      xDiffs[0] * interpolant.x + uvs[3],
      xDiffs[1] * interpolant.x + uvs[0]
    };
    const vector2_type yDiff = yVals[1] - yVals[0];
    vector2_type uv = yDiff * interpolant.y + yVals[0];

    cache.xDiffs[0] = xDiffs[0];
    cache.xDiffs[1] = xDiffs[1];
    cache.yDiff = yDiff;
    cache.warpedUv = uv;
    cache.interpolantY = interpolant.y;

    const codomain_type result = _postWarp.generate(uv, cache.postWarpCache);

    return result;
  }

  density_type forwardPdf(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    const scalar_type detInterpolJacobian = determinant(matrix<scalar_type, 2, 2>(
      lerp(cache.xDiffs[0], cache.xDiffs[1], cache.interpolantY), // first column dFdx
      cache.yDiff // second column dFdy
    )) * scalar_type(_lastTexel.x) * scalar_type(_lastTexel.y);
    const scalar_type pdf = abs(_postWarp.forwardPdf(cache.warpedUv, cache.postWarpCache) / detInterpolJacobian);
    return pdf;
  }

  weight_type forwardWeight(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    vector2_type texelCoord = cache.warpedUv * vector2_type(_lastTexel.x, _lastTexel.y);
    float32_t2 lumaMapUv = (texelCoord + float32_t2(0.5, 0.5)) / vector2_type(_lastTexel.x + 1, _lastTexel.y + 1);
    weight_type luma;
    _lumaMap.get(cache.warpedUv, luma);
    return (luma * _postWarp.forwardWeight(cache.warpedUv, cache.postWarpCache)) / _lumaMap.getAvgLuma();
  }

  weight_type backwardWeight(codomain_type direction) NBL_CONST_MEMBER_FUNC
  {
    vector2_type envmapUv = _postWarp.generateInverse(direction);
    vector2_type texelCoord = envmapUv * vector2_type(_lastTexel.x, _lastTexel.y);
    float32_t2 lumaMapUv = (texelCoord + float32_t2(0.5, 0.5)) / vector2_type(_lastTexel.x + 1, _lastTexel.y + 1);
    scalar_type luma;
    _lumaMap.get(lumaMapUv, luma);
    return (luma * _postWarp.backwardWeight(direction)) / _lumaMap.getAvgLuma();
  }
};

}
}
}

#endif



