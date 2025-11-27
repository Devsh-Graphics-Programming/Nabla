#pragma shader_stage(vertex)

#include "nbl/ext/DebugDraw/builtin/hlsl/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::debug_draw;

[[vk::push_constant]] SPushConstants pc;

[shader("vertex")]
PSInput main()
{
    PSInput output;
    const float32_t3 vertex = getUnitAABBVertex();
    InstanceData instance = vk::BufferPointer<InstanceData>(pc.pInstanceBuffer + sizeof(InstanceData) * glsl::gl_InstanceIndex()).Get();

    output.position = math::linalg::promoted_mul(instance.transform, vertex);
    output.color = instance.color;

    return output;
}