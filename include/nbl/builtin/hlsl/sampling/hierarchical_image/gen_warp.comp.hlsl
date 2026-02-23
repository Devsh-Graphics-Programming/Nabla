#include "nbl/builtin/hlsl/sampling/hierarchical_image.hlsl"

[[vk::binding(0, 0)]] Texture2D<float32_t> lumaMap;

[[vk::binding(1, 0)]] RWTexture2D<float32_t2> outImage;

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling;

struct LuminanceAccessor
{
    float32_t texelFetch(uint32_t2 coord, uint32_t level)
    {
        return lumaMap.Load(uint32_t3(coord, level));
    }

    float32_t4 texelGather(uint32_t2 coord, uint32_t level)
    {
        return float32_t4(
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 1)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 1)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(1, 0)),
            lumaMap.Load(uint32_t3(coord, level), uint32_t2(0, 0))
        );

    }
};

[numthreads(WORKGROUP_DIM, WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
    LuminanceAccessor luminanceAccessor;
    uint32_t lumaMapWidth, lumaMapHeight;

    lumaMap.GetDimensions(lumaMapWidth, lumaMapHeight);

    using LuminanceSampler = LuminanceMapSampler<float32_t, LuminanceAccessor>;

    LuminanceSampler luminanceSampler = 
      LuminanceSampler::create(luminanceAccessor, uint32_t2(lumaMapWidth, lumaMapHeight), lumaMapWidth != lumaMapHeight, uint32_t2(lumaMapWidth, lumaMapHeight));

    uint32_t2 pixelCoord = threadID.xy;

    outImage[pixelCoord] = luminanceSampler.binarySearch(pixelCoord);

}
