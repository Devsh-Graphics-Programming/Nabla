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
    float32_t load(uint32_t2 coord, uint32_t level) NBL_CONST_MEMBER_FUNC
    {
        assert(coord.x < pc.warpMapWidth && coord.y < pc.warpMapHeight);
        return lumaMap.Load(uint32_t3(coord, level));
    }

};

[numthreads(GEN_WARP_WORKGROUP_DIM, GEN_WARP_WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{
  if (threadID.x < pc.warpMapWidth && threadID.y < pc.warpMapHeight)
  {
    using WarpGenerator = HierarchicalWarpGenerator<float32_t, LuminanceAccessor>;

    const LuminanceAccessor luminanceAccessor;

    const WarpGenerator warpGenerator = WarpGenerator::create(luminanceAccessor, uint32_t2(pc.lumaMapWidth, pc.lumaMapHeight), pc.lumaMapWidth != pc.lumaMapHeight);

    const uint32_t2 pixelCoord = threadID.xy;

    const float32_t2 xi = float32_t2(pixelCoord) / float32_t2(pc.warpMapWidth - 1, pc.warpMapHeight - 1);

    outImage[pixelCoord] = warpGenerator.generate(xi).value();
  }


}
