
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace envmap
{
namespace impl 
{
  bool choseSecond(float first, float second, NBL_REF_ARG(float) xi)
  {
    // numerical resilience against IEEE754
    float firstProb = 1.0f / (1.0f + second / first);
    float dummy = 0.0f;
    return math::partitionRandVariable(firstProb, xi, dummy);
  }

}

}
}
}
}

#ifdef __HLSL_VERSION
namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace envmap
{

struct WarpmapGeneration 
{

  template <typename LuminanceAccessor, typename OutputAccessor NBL_FUNC_REQUIRES (envmap::LuminanceReadAccessor<LuminanceAccessor> && envmap::WarpmapWriteAccessor<OutputAccessor>)
  // TODO(kevinyu): Should lumapMapSize and warpMapSize provided by Accessor?
  static void __call(NBL_CONST_REF_ARG(LuminanceAccessor) luminanceAccessor, NBL_REF_ARG(OutputAcessor) outputAccessor, uint32_t2 lumaMapSize, uint32_t2 warpMapSize)
  {
    const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
    const uint32_t lastWarpMapPixel = warpMapSize - uint32_t2(1, 1);

    if (all(threadID < warpMapSize))
    {
      float32_t2 xi = float32_t2(threadID) / float32_t2(lastWarpMapPixel);

      uint32_t2 p;
      p.y = 0;

      // TODO(kevinyu): Implement findMSB
      const uint32_t2 mip2x1 = findMSB(lumaMapSize.x) - 1;
		  // do one split in the X axis first cause penultimate full mip would have been 2x1
      p.x = impl::choseSecond(luminanceAccessor.get(uint32_t2(0, 0), mip2x1, uint32_t2(0, 0)), luminanceAccessor.get(uint32_t2(0, 0), mip2x1, uint32_t2(1, 0), xi.x) ? 1 : 0;
      for (uint32_t i = mip2x1; i != 0;)
      {
        --i;
        p <<= 1;
        const float32_t4 values = float32_t4(
          luminanceAccessor.get(p, i, uint32_t2(0, 1)),
          luminanceAccessor.get(p, i, uint32_t2(1, 1)),
          luminanceAccessor.get(p, i, uint32_t2(1, 0)),
          luminanceAccessor.get(p, i, uint32_t2(0, 0))
        );

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
        {
          p.x |= 1;
        }
      }

      const float32_t2 directionUV = (float32_t2(p.x, p.y) + xi) / float32_t2(lumaMapSize);
      outputAccessor.set(threadID, directionUV);
    }
  }

};

}
}
}
}
#endif

#endif