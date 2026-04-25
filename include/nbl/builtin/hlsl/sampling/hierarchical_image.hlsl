// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_

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

// TODO: Implement corner sampling or centered sampling based on the type of LuminanceAccessor
template <typename LuminanceAccessorT 
  NBL_PRIMARY_REQUIRES(
    hierarchical_image::MipmappedLuminanceReadAccessor<LuminanceAccessorT>
  )
struct HierarchicalLuminanceSampler
{
  using this_type = HierarchicalLuminanceSampler<LuminanceAccessorT>;
  using scalar_type = typename LuminanceAccessorT::value_type;
  using vector2_type = vector<scalar_type, 2>;
  using vector4_type = vector<scalar_type, 4>;
  using domain_type = vector2_type;
  using codomain_type = vector2_type;
  using weight_type = scalar_type;
  using density_type = scalar_type;
  struct cache_type
  {
    scalar_type rcpPmf;
  };

  LuminanceAccessorT _map;
  uint16_t2 _lastTexel;
  uint16_t _lastMipLevel : 15;
  uint16_t _aspect2x1 : 1;

  static this_type create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap)
  {
    this_type result;
    result._map = lumaMap;
    const uint16_t2 mapSize = lumaMap.resolution();
    result._lastTexel = mapSize - uint16_t2(1, 1);
    // Note: We use mapSize.y here because currently the map aspect ratio can only be 1x1 or 2x1
    result._lastMipLevel = _static_cast<uint16_t>(findMSB(_static_cast<uint32_t>(mapSize.y)));
    result._aspect2x1 = mapSize.x != mapSize.y;
    return result;
  }

  static bool __choseSecond(scalar_type first, scalar_type second, NBL_REF_ARG(scalar_type) xi, NBL_REF_ARG(scalar_type) rcpPmf)
  {
    scalar_type rcpChoiceProb = scalar_type(0);
    PartitionRandVariable<scalar_type> partition;
    // Why not simply first / (first + second)? Because `(1 / (1 + second / first))` is equivalent and more numerical resilient against IEEE754. if both first and second are very large then first + second overflows to inf. By computing second/ first, we get a finite ratio and 1 + ratio stays representable.
    partition.leftProb = scalar_type(1) / (scalar_type(1) + (second / first));
    bool choseSecond = partition(xi, rcpChoiceProb);
    rcpPmf *= rcpChoiceProb;
    return choseSecond;
  }

  // Cannot use textureGather since we need to pass the mipLevel
  vector4_type __texelGather(uint16_t2 coord, uint16_t level) NBL_CONST_MEMBER_FUNC
  {
    assert(coord.x < _lastTexel.x && coord.y < _lastTexel.y);
    scalar_type p0, p1, p2, p3;
    _map.get(p0, coord + uint16_t2(0, 1), level);
    _map.get(p1, coord + uint16_t2(1, 1), level);
    _map.get(p2, coord + uint16_t2(1, 0), level);
    _map.get(p3, coord + uint16_t2(0, 0), level);
    return vector4_type(p0, p1, p2, p3);
  }

  codomain_type generate(const domain_type v, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
  {
    uint16_t2 p = uint16_t2(0, 0);

    domain_type xi = v;
    scalar_type rcpPmf = 1.0;
    if (_aspect2x1) {
      scalar_type p0, p1;
      // do one split in the X axis first cause penultimate full mip would have been 2x1
      _map.get(p0, uint16_t2(0, 0), _lastMipLevel);
      _map.get(p1, uint16_t2(1, 0), _lastMipLevel);
      p.x = __choseSecond(p0, p1, xi.x, rcpPmf) ? 1 : 0;
    }

    for (int i = _lastMipLevel - 1; i >= 0; i--)
    {
      p <<= 1;
      const vector4_type values = __texelGather(p, i);
      scalar_type wx_0, wx_1;
      {
        const scalar_type wy_0 = values[3] + values[2];
        const scalar_type wy_1 = values[1] + values[0];
        if (__choseSecond(wy_0, wy_1, xi.y, rcpPmf))
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
      if (__choseSecond(wx_0, wx_1, xi.x, rcpPmf))
        p.x |= 1;
    }


    // If we don`t add xi, the sample will clump to the lowest corner of environment map texel. Each time we call PartitionRandVariable(), the output xi is the new xi that determines how left and right(or top and bottom for y axis) to choose the child partition. It means that if for some input xi, the output xi = 0, then the input xi is the edge of choosing this partition and the previous partition, and vice versa, if output xi = 1, then the input xi is the edge of choosing this partition and the next partition. Hence, by adding xi to the lower corner of the texel, we create a gradual transition from one pixel to another. Without adding output xi, the calculation of jacobian using the difference of sample value would not work.
    // Since we want to do corner sampling. We have to handle edge texels as corner cases. Remember, in corner sampling we map uv [0,1] to [center of first texel, center of last texel]. So when p is an edge texel, we have to remap xi. [0.0, 0.5] when p == 0, and [0.5, 1] when p == length - 1.
    if (p.x == 0)
      xi.x = xi.x * scalar_type(0.5) + scalar_type(0.5);
    if (p.y == 0)
      xi.y = xi.y * scalar_type(0.5) + scalar_type(0.5);
    if (p.x == _lastTexel.x)
      xi.x = xi.x * scalar_type(0.5);
    if (p.y == _lastTexel.y)
      xi.y = xi.y * scalar_type(0.5);

    // We reduce by 0.5 and divide with _lastTexel instead of map size to normalize the cornered sampling coordinate
    const vector2_type directionUV = (vector2_type(p.x, p.y) + xi - domain_type(0.5, 0.5)) / _lastTexel;

    cache.rcpPmf = rcpPmf;

    return directionUV;
  }

  density_type forwardPdf(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    return density_type(_lastTexel.x) * density_type(_lastTexel.y) / cache.rcpPmf;
  }

  weight_type forwardWeight(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    return forwardPdf(xi, cache);
  }

  density_type backwardPdf(const codomain_type codomainVal) NBL_CONST_MEMBER_FUNC
  {
    vector2_type texelCoord = codomainVal * vector2_type(_lastTexel.x, _lastTexel.y);
    float32_t2 lumaMapUv = (texelCoord + float32_t2(0.5, 0.5)) / vector2_type(_lastTexel.x + 1, _lastTexel.y + 1);
    weight_type luma;
    _map.get(lumaMapUv, luma);
    return luma / _map.getAvgLuma();
  }

  weight_type backwardWeight(const codomain_type codomainVal) NBL_CONST_MEMBER_FUNC
  {
    return backwardPdf(codomainVal);
  }

};

// TODO(kevinyu): Add constraint for PostWarpT
template <typename LuminanceAccessorT, typename PostWarpT 
  NBL_PRIMARY_REQUIRES(
    hierarchical_image::MipmappedLuminanceReadAccessor<LuminanceAccessorT> && 
    concepts::BackwardTractableSampler<PostWarpT>
  )
struct ComposedHierarchicalSampler
{
  using this_type = ComposedHierarchicalSampler<LuminanceAccessorT, PostWarpT>;
  using warp_generator_type = HierarchicalLuminanceSampler<LuminanceAccessorT>;
  using scalar_type = typename LuminanceAccessorT::value_type;
  using density_type = scalar_type;
  using weight_type = scalar_type;
  using vector2_type = vector<scalar_type, 2>;
  using vector3_type = vector<scalar_type, 3>;
  using vector4_type = vector<scalar_type, 4>;
  using domain_type = typename warp_generator_type::domain_type;
  using codomain_type = typename PostWarpT::codomain_type;

  static_assert(is_same_v<typename PostWarpT::domain_type, typename warp_generator_type::codomain_type> && is_same_v<typename PostWarpT::density_type, density_type> && is_same_v<typename PostWarpT::weight_type, weight_type>);

  struct cache_type
  {
    typename warp_generator_type::cache_type warpGeneratorCache;
    typename PostWarpT::cache_type postWarpCache;
    typename PostWarpT::domain_type warpSample;
  };

  warp_generator_type _warpGenerator;
  PostWarpT _postWarp;

  static this_type create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap)
  {
    this_type result;
    result._warpGenerator = warp_generator_type::create(lumaMap);
    return result;
  }

  codomain_type generate(const domain_type xi, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
  {
    const typename warp_generator_type::codomain_type warpSample = _warpGenerator.generate(xi, cache.warpGeneratorCache);
    const codomain_type postWarpSample = _postWarp.generate(warpSample, cache.postWarpCache);
    cache.warpSample = warpSample;
    
    return postWarpSample;
  }

  density_type forwardPdf(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    return _warpGenerator.forwardPdf(xi, cache.warpGeneratorCache) * _postWarp.forwardPdf(cache.warpSample, cache.postWarpCache);
  }

  weight_type forwardWeight(const domain_type xi, const cache_type cache) NBL_CONST_MEMBER_FUNC
  {
    return  _warpGenerator.forwardWeight(xi, cache.warpGeneratorCache) * _postWarp.forwardWeight(cache.warpSample, cache.postWarpCache);
  }

  density_type backwardPdf(const codomain_type codomainVal) NBL_CONST_MEMBER_FUNC
  {
    typename PostWarpT::domain_type postWarpDomain = _postWarp.generateInverse(codomainVal);
    return _warpGenerator.backwardPdf(postWarpDomain) * _postWarp.backwardPdf(codomainVal);
  }

  weight_type backwardWeight(const codomain_type codomainVal) NBL_CONST_MEMBER_FUNC
  {
    typename PostWarpT::domain_type postWarpDomain = _postWarp.generateInverse(codomainVal);
    return _warpGenerator.backwardWeight(postWarpDomain) * _postWarp.backwardWeight(codomainVal);
  }

};



}
}
}

#endif
