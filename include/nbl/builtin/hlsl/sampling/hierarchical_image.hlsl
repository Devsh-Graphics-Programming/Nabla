// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_

#include <nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl>
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

// TODO(kevinyu): Temporary struct before PR #1001 merged to master
template<typename V, typename P>
struct value_and_rcpPdf
{
  using this_t = value_and_rcpPdf<V, P>;

  static this_t create(const V _value, const P _rcpPdf)
  {
    this_t retval;
    retval._value = _value;
    retval._rcpPdf = _rcpPdf;
    return retval;
  }

  V value() { return _value; }
  P rcpPdf() { return _rcpPdf; }

  V _value;
  P _rcpPdf;
};

template<typename V, typename P>
struct value_and_pdf
{
  using this_t = value_and_pdf<V, P>;

  static this_t create(const V _value, const P _pdf)
  {
    this_t retval;
    retval._value = _value;
    retval._pdf = _pdf;
    return retval;
  }

  V value() { return _value; }
  P pdf() { return _pdf; }

  V _value;
  P _pdf;
};

// TODO: Add an option for corner sampling or centered sampling as boolean parameter
template <typename ScalarT, typename LuminanceAccessorT 
  NBL_PRIMARY_REQUIRES(
    is_scalar_v<ScalarT> && 
    concepts::accessors::MipmappedLoadableImage<LuminanceAccessorT, ScalarT, 2, 1>
  )
struct HierarchicalWarpGenerator
{
  using scalar_type = ScalarT;
  using vector2_type = vector<scalar_type, 2>;
  using vector4_type = vector<scalar_type, 4>;
  using domain_type = vector2_type;
  using codomain_type = vector2_type;
  using sample_type = value_and_pdf<codomain_type, scalar_type>;
  using density_type = scalar_type;

  LuminanceAccessorT _map;
  uint16_t2 _mapSize;
  uint16_t _layerIndex;
  uint16_t _lastMipLevel : 15;
  uint16_t _aspect2x1 : 1;

  static HierarchicalWarpGenerator<ScalarT, LuminanceAccessorT> create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, uint16_t2 mapSize, uint16_t layerIndex)
  {
    HierarchicalWarpGenerator<ScalarT, LuminanceAccessorT> result;
    result._map = lumaMap;
    result._mapSize = mapSize;
    result._layerIndex = layerIndex;
    // Note: We use mapSize.y here because the currently the map aspect ratio can only be 1x1 or 2x1
    result._lastMipLevel = _static_cast<uint16_t>(findMSB(_static_cast<uint32_t>(mapSize.y)));
    result._aspect2x1 = mapSize.x != mapSize.y;
    return result;
  }

  static bool __choseSecond(scalar_type first, scalar_type second, NBL_REF_ARG(scalar_type) xi, NBL_REF_ARG(scalar_type) rcpPmf) NBL_CONST_MEMBER_FUNC
  {
    // numerical resilience against IEEE754
    scalar_type rcpChoiceProb = scalar_type(0);
    PartitionRandVariable<scalar_type> partition;
    partition.leftProb = scalar_type(1) / (scalar_type(1) + (second / first));
    bool choseSecond = partition(xi, rcpChoiceProb);
    rcpPmf *= rcpChoiceProb;
    return choseSecond;
  }

  // Cannot use textureGather since we need to pass the mipLevel
  vector4_type __texelGather(uint16_t2 coord, uint16_t level) NBL_CONST_MEMBER_FUNC
  {
    assert(coord.x < _mapSize.x - 1 && coord.y < _mapSize.y - 1);
    vector<scalar_type, 1> p0, p1, p2, p3;
    _map.get(p0, coord + uint16_t2(0, 1), _layerIndex, level);
    _map.get(p1, coord + uint16_t2(1, 1), _layerIndex, level);
    _map.get(p2, coord + uint16_t2(1, 0), _layerIndex, level);
    _map.get(p3, coord + uint16_t2(0, 0), _layerIndex, level);
    return vector4_type(p0, p1, p2, p3);
  }

  sample_type generate(vector2_type xi) NBL_CONST_MEMBER_FUNC
  {
    uint16_t2 p = uint16_t2(0, 0);

    scalar_type rcpPmf = 1;
    if (_aspect2x1) {
      vector<scalar_type, 1> p0, p1;
      // do one split in the X axis first cause penultimate full mip would have been 2x1
      _map.get(p0, uint16_t2(0, 0), _layerIndex, _lastMipLevel);
      _map.get(p1, uint16_t2(1, 0), _layerIndex, _lastMipLevel);
      p.x = __choseSecond(p0.x, p1, xi.x, rcpPmf) ? 1 : 0;
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
    return sample_type::create(directionUV, (_mapSize.x * _mapSize.y) / rcpPmf);
  }

  density_type forwardPdf(domain_type xi) NBL_CONST_MEMBER_FUNC
  {
    return generate(xi).pdf();
  }

  // Doesn't comply with sampler concept. This class is extracted so can be used on warpmap generation without passing in unnecessary information like avgLuma. So, need to pass in avgLuma when calculating backwardPdf.
  density_type backwardPdf(codomain_type codomainVal, scalar_type rcpAvgLuma) NBL_CONST_MEMBER_FUNC
  {
    return _map.load(codomainVal) * rcpAvgLuma;
  }

};

template <typename ScalarT, typename LuminanceAccessorT, typename PostWarpT 
  NBL_PRIMARY_REQUIRES(
    is_scalar_v<ScalarT> && 
    concepts::accessors::MipmappedLoadableImage<LuminanceAccessorT, ScalarT, 2, 1> &&
    concepts::Warp<PostWarpT>
  )
struct HierarchicalWarpSampler
{
  using warp_generator_type = HierarchicalWarpGenerator<ScalarT, LuminanceAccessorT>;
  using warp_sample_type = typename warp_generator_type::sample_type;
  using scalar_type = ScalarT;
  using density_type = scalar_type;
  using vector2_type = vector<scalar_type, 2>;
  using vector3_type = vector<scalar_type, 3>;
  using vector4_type = vector<scalar_type, 4>;
  using domain_type = vector2_type;
  using codomain_type = vector3_type;
  using sample_type = value_and_pdf<codomain_type, density_type>;
  
  warp_generator_type _warpGenerator;
  scalar_type _rcpAvgLuma;

  static HierarchicalWarpSampler<ScalarT, LuminanceAccessorT, PostWarpT> create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, scalar_type avgLuma, uint16_t2 mapSize, uint16_t layerIndex)
  {
    HierarchicalWarpSampler result;
    result._warpGenerator = warp_generator_type::create(lumaMap, mapSize, layerIndex);
    result._rcpAvgLuma = scalar_type(1.0) / avgLuma;
    return result;
  }

  sample_type generate(domain_type xi) NBL_CONST_MEMBER_FUNC
  {
    const warp_sample_type warpSample = _warpGenerator.generate(xi);
    const WarpResult<codomain_type> postWarpResult = PostWarpT::warp(warpSample.value());
    return sample_type::create(postWarpResult.dst, postWarpResult.density * warpSample.pdf());
  }

  density_type forwardPdf(domain_type xi) NBL_CONST_MEMBER_FUNC
  {
    const warp_sample_type warpSample = _warpGenerator.generate(xi);
    return PostWarpT::forwardDensity(warpSample.value()) * warpSample.pdf();
  }

  density_type backwardPdf(codomain_type codomainVal) NBL_CONST_MEMBER_FUNC
  {
    return PostWarpT::backwardPdf(codomainVal, _rcpAvgLuma) * _warpGenerator.backwardPdf(codomainVal);
  }

};


template <typename ScalarT, typename LuminanceAccessorT, typename HierarchicalSamplerT, typename PostWarpT 
  NBL_PRIMARY_REQUIRES(is_scalar_v<ScalarT> &&
    concepts::accessors::GenericReadAccessor<LuminanceAccessorT, ScalarT, float32_t2> &&
    hierarchical_image::WarpAccessor<HierarchicalSamplerT, ScalarT> &&
    concepts::Warp<PostWarpT>)
struct WarpmapSampler 
{
  using scalar_type = ScalarT;
  using vector2_type = vector<ScalarT, 2>;
  using vector3_type = vector<ScalarT, 3>;
  using vector4_type = vector<ScalarT, 4>;
  using domain_type = vector2_type;
  using codomain_type = vector3_type;
  using weight_type = scalar_type;
  using sample_type = value_and_pdf<codomain_type, weight_type>;

  LuminanceAccessorT _lumaMap;
  HierarchicalSamplerT _warpMap;
  uint32_t _effectiveWarpArea;
  scalar_type _rcpAvgLuma;

  static WarpmapSampler create(NBL_CONST_REF_ARG(LuminanceAccessorT) lumaMap, NBL_CONST_REF_ARG(HierarchicalSamplerT) warpMap, uint16_t2 warpSize, scalar_type avgLuma) 
  {
    WarpmapSampler<ScalarT, LuminanceAccessorT, HierarchicalSamplerT, PostWarpT> result;
    result._lumaMap = lumaMap;
    result._warpMap = warpMap;
    result._effectiveWarpArea = (warpSize.x - 1) * (warpSize.y - 1);
    result._rcpAvgLuma = ScalarT(1.0) / avgLuma;
    return result;
  }

  weight_type forwardWeight(domain_type xi) NBL_CONST_MEMBER_FUNC
  {
    return generate(xi).value();
  }

  weight_type backwardWeight(codomain_type direction) NBL_CONST_MEMBER_FUNC
  {
    vector2_type envmapUv = PostWarpT::inverseWarp(direction);
    scalar_type luma;
    _lumaMap.get(envmapUv, luma);
    return luma * _rcpAvgLuma * PostWarpT::backwardDensity(direction);
  }

  sample_type generate(vector2_type xi) NBL_CONST_MEMBER_FUNC
  {
    const vector2_type interpolant;
    matrix<scalar_type, 4, 2> uvs; 
    _warpMap.gatherUv(xi, uvs, interpolant);

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

    const WarpResult<vector3_type> warpResult = PostWarpT::warp(uv);

    const scalar_type detInterpolJacobian = determinant(matrix<scalar_type, 2, 2>(
      lerp(xDiffs[0], xDiffs[1], interpolant.y), // first column dFdx
      yDiff // second column dFdy
    )) * _effectiveWarpArea;

    const scalar_type pdf = abs(warpResult.density / detInterpolJacobian);

    return sample_type::create(warpResult.dst, pdf);
  }
};

}
}
}

#endif
