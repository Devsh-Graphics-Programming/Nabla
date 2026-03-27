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
    template <typename T, int32_t Dims
      NBL_FUNC_REQUIRES(concepts::same_as<T, float32_t> && Dims == 2)
    void get(NBL_REF_ARG(vector<T, 1>) outVal, vector<uint16_t, Dims> pixelCoord, uint16_t layer, uint16_t level) NBL_CONST_MEMBER_FUNC
    {
        assert(pixelCoord.x < pc.warpMapWidth && pixelCoord.y < pc.warpMapHeight);
        outVal = lumaMap.Load(int4(pixelCoord, layer, level));
    }

};

[numthreads(GenWarpWorkgroupDim, GenWarpWorkgroupDim, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  if (all(threadID.xyz < uint32_t3(pc.warpMapHeight, pc.warpMapWidth, pc.lumaMapLayer)))
  {
    using WarpGenerator = HierarchicalWarpGenerator<float32_t, LuminanceAccessor>;

    const LuminanceAccessor luminanceAccessor;

    const WarpGenerator warpGenerator = WarpGenerator::create(luminanceAccessor, uint32_t2(pc.lumaMapWidth, pc.lumaMapHeight), pc.lumaMapWidth != pc.lumaMapHeight);

    const uint32_t2 pixelCoord = threadID.xy;

    const float32_t2 xi = float32_t2(pixelCoord) / float32_t2(pc.warpMapWidth - 1, pc.warpMapHeight - 1);

    outImage[threadID.xyz] = warpGenerator.generate(xi).value();
  }


}
