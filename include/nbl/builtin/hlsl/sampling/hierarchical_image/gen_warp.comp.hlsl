#include "nbl/builtin/hlsl/sampling/hierarchical_image.hlsl"
#include "nbl/builtin/hlsl/sampling/hierarchical_image/common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling;
using namespace nbl::hlsl::sampling::hierarchical_image;

[[vk::push_constant]] SWarpGenPushConstants pc;

[[vk::binding(0, 0)]] Texture2DArray<float32_t> lumaMap;
[[vk::binding(1, 0)]] RWTexture2DArray<float32_t2> outImage;


struct LuminanceAccessor
{
    using value_type = float32_t;

    uint16_t _layerIndex;

    static LuminanceAccessor create(uint16_t layerIndex)
    {
        LuminanceAccessor result;
        result._layerIndex = layerIndex;
        return result;
    }

    void get(NBL_REF_ARG(value_type) outVal, uint16_t2 pixelCoord, uint16_t level) NBL_CONST_MEMBER_FUNC
    {
        assert(pixelCoord.x < pc.warpMapWidth && pixelCoord.y < pc.warpMapHeight);
        outVal = lumaMap.Load(int4(pixelCoord, _layerIndex, level));
    }

    uint16_t2 resolution() NBL_CONST_MEMBER_FUNC
    {
        return uint16_t2(pc.lumaMapWidth, pc.lumaMapHeight);
    }

    value_type getAvgLuma() NBL_CONST_MEMBER_FUNC
    {
        const uint16_t lastMipLevel = _static_cast<uint16_t>(findMSB(_static_cast<uint32_t>(pc.warpMapHeight)));
        if (pc.warpMapHeight == pc.warpMapWidth)
        {
            return lumaMap.Load(int4(0, 0, _layerIndex, lastMipLevel));
        } else
        {
            return value_type(0.5) * (lumaMap.Load(int4(0, 0, _layerIndex, lastMipLevel)) + lumaMap.Load(int4(1, 0, _layerIndex, lastMipLevel)));
        }
    }

};

[numthreads(GenWarpWorkgroupDim, GenWarpWorkgroupDim, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  if (all(threadID.xyz < uint32_t3(pc.warpMapHeight, pc.warpMapWidth, pc.lumaMapLayer)))
  {
    using WarpGenerator = HierarchicalLuminanceSampler<LuminanceAccessor>;

    const uint16_t layerIndex = threadID.z;
    const LuminanceAccessor luminanceAccessor = LuminanceAccessor::create(layerIndex);

    const WarpGenerator warpGenerator = WarpGenerator::create(luminanceAccessor);

    const uint32_t2 pixelCoord = threadID.xy;

    const float32_t2 xi = float32_t2(pixelCoord) / float32_t2(pc.warpMapWidth - 1, pc.warpMapHeight - 1);

    typename WarpGenerator::cache_type dummyCache;
    outImage[threadID.xyz] = warpGenerator.generate(xi, dummyCache);
  }


}
