// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_HIERARCHICAL_IMAGE_INCLUDED_

#include <nbl/builtin/hlsl/sampling/warp.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/hierarchical_image.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{
template <typename T, typename LuminanceAccessor, typename PostWarp NBL_PRIMARY_REQUIRES(is_scalar_v<T> && hierarchical_image::LuminanceReadAccessor<LuminanceAccessor> && Warp<PostWarp>)
struct HierarchicalImage 
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;
    using vector4_type = vector<T, 4>;
    LuminanceAccessor accessor;
    uint32_t2 lumaMapSize;
    bool lumaAspect2x1;
    uint32_t2 lastWarpPixel;

    static vector2_type calculateSampleAndPdf(NBL_REF_ARG(scalar_type) rcpPdf, vector4_type dirsX, vector4_type dirsY, vector2_type unnormCoord, uint32_t2 lastWarpPixel)
    {
	  // TODO(kevinyu): Convert float32_t to scalar_type
      const float32_t2 interpolant = frac(unnormCoord);
      const float32_t4x2 uvs = transpose(float32_t2x4(dirsX, dirsY));

      const float32_t2 xDiffs[] = {
        uvs[2] - uvs[3],
        uvs[1] - uvs[0]
      };
      const float32_t2 yVals[] = {
        xDiffs[0] * interpolant.x + uvs[3],
        xDiffs[1] * interpolant.x + uvs[0]
      };
      const float32_t2 yDiff = yVals[1] - yVals[0];
      const float32_t2 uv = yDiff * interpolant.y + yVals[0];

      // Note(kevinyu): sinTheta is calculated twice inside PostWarp::warp and PostWarp::forwardDensity
      const float32_t3 L = PostWarp::warp(uv);

      const float detInterpolJacobian = determinant(float32_t2x2(
        lerp(xDiffs[0], xDiffs[1], interpolant.y), // first column dFdx
        yDiff // second column dFdy
      ));

      rcpPdf = abs((detInterpolJacobian * scalar_t(lastWarpPixel.x * lastWarpPixel.y) / PostWarp::forwardDensity(uv));

      return L;
    }

    static HierarchicalImage create(NBL_CONST_REF_ARG(LuminanceAccessor) accessor, const uint32_t2 lumaMapSize, const bool lumaAspect2x1, const uint32_t2 warpSize) 
    {
	    HierarchicalImage<T, LuminanceAccessor, PostWarp> result;
        result.accessor = accessor;
        result.lumaMapSize = lumaMapSize;
        result.lumaAspect2x1 = lumaAspect2x1;
        result.lastWarpPixel = warpSize - uint32_t2(1, 1);
        return result;
    }

	static vector<scalar_type, 2> binarySearch(const vector<scalar_type, 2> xi)
    {
      uint32_t2 p = uint32_t2(0, 0);

      if (aspect2x1) {
        // TODO(kevinyu): Implement findMSB
        const uint32_t2 mip2x1 = findMSB(lumaMapSize.x) - 1;

        // do one split in the X axis first cause penultimate full mip would have been 2x1
        p.x = impl::choseSecond(luminanceAccessor.fetch(uint32_t2(0, 0), mip2x1), luminanceAccessor.fetch(uint32_t2(0, 1), mip2x1), xi.x) ? 1 : 0;
      }

      for (uint32_t i = mip2x1; i != 0;)
      {
        --i;
        p <<= 1;
        const float32_t4 values = luminanceAccessor.gather(p, i);
        float32_t wx_0, wx_1;
        {
          const float32_t wy_0 = values[3] + values[2];
          const float32_t wy_1 = values[1] + values[0];
          if (impl::choseSecond(wy_0, wy_1, xi.y))
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

        if (impl::choseSecond(wx_0, wx_1, xi.x))
          p.x |= 1;
      }

      // TODO(kevinyu): Add some comment why we add xi.
      const float32_t2 directionUV = (float32_t2(p.x, p.y) + xi) / float32_t2(lumaMapSize);
      return directionUV;
    }

    uint32_t2 generate(NBL_REF_ARG(scalar_type) rcpPdf, vector<scalar_type, 2> xi)
	{
      const float32_t2 unnormCoord = xi * lastWarpPixel;
      const float32_t2 warpSampleCoord = (unnormCoord + float32_t2(0.5f, 0.5f)) / float32_t2(warpmapSize.x, warpmapSize.y);
      const float32_t2 dir0 = binarySearch(luminanceMap, lumaMapSize, warpSampleCoord + float32_t2(0, 1), lumaAspect2x1);
      const float32_t2 dir1 = binarySearch(luminanceMap, lumaMapSize, warpSampleCoord + float32_t2(1, 1), lumaAspect2x1);
      const float32_t2 dir2 = binarySearch(luminanceMap, lumaMapSize, warpSampleCoord + float32_t2(1, 0), lumaAspect2x1);
      const float32_t2 dir3 = binarySearch(luminanceMap, lumaMapSize, warpSampleCoord, lumaAspect2x1);

      const float32_t4 dirsX = float32_t4(dir0.x, dir1.x, dir2.x, dir3.x);
      const float32_t4 dirsY = float32_t4(dir1.y, dir1.y, dir2.y, dir3.y);

      return calculateSampleAndPdf(rcpPdf, dirsX, dirsY, unnormCoord, lastWarpPixel);
	}
};

//TODO(kevinyu): Impelemnt cached warp map sampler

}
}

#endif
