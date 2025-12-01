#pragma shader_stage(vertex)

#include "nbl/ext/DebugDraw/builtin/hlsl/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::debug_draw;

[[vk::push_constant]] SSinglePushConstants pc;

[shader("vertex")]
PSInput main()
{
    PSInput output;
    float32_t3 vertex = getUnitAABBVertex();

    output.position = math::linalg::promoted_mul(pc.instance.transform, vertex);
    output.color = pc.instance.color;

    return output;
}