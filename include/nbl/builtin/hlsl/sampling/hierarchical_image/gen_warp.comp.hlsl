#include "nbl/builtin/hlsl/sampling/hierarchical_image.hlsl"
#include "nbl/builtin/hlsl/sampling/hierarchical_image/common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling;
using namespace nbl::hlsl::sampling::hierarchical_image;

[[vk::push_constant]] SWarpGenPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float32_t> lumaMap;

[[vk::binding(1, 0)]] RWTexture2D<float32_t2> outImage;


struct LuminanceAccessor
{
    float32_t texelFetch(uint32_t2 coord, uint32_t level)
    {
        assert(coord.x < pc.warpMapWidth && coord.y < pc.warpMapHeight);
        return lumaMap.Load(uint32_t3(coord, level));
    }

    float32_t4 texelGather(uint32_t2 coord, uint32_t level)
    {
        assert(coord.x < pc.warpMapWidth - 1 && coord.y < pc.warpMapHeight - 1);
        return float32_t4(
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 1)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 1)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 0)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 0))
        );
    }

};

[numthreads(GEN_WARP_WORKGROUP_DIM, GEN_WARP_WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  if (threadID.x < pc.warpMapWidth && threadID.y < pc.warpMapHeight)
  {
    using LuminanceSampler = LuminanceMapSampler<float32_t, LuminanceAccessor>;

    LuminanceAccessor luminanceAccessor;
    LuminanceSampler luminanceSampler = 
      LuminanceSampler::create(luminanceAccessor, uint32_t2(pc.lumaMapWidth, pc.lumaMapHeight), pc.lumaMapWidth != pc.lumaMapHeight, uint32_t2(pc.warpMapWidth, pc.warpMapHeight));

    uint32_t2 pixelCoord = threadID.xy;

    outImage[pixelCoord] = luminanceSampler.binarySearch(pixelCoord);
  }


}
