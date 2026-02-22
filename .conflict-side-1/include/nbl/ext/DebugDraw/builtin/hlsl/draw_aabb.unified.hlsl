#include "nbl/ext/DebugDraw/builtin/hlsl/common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::debug_draw;

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
PSInput aabb_vertex_single()
{
    PSInput output;
    float32_t3 vertex = getUnitAABBVertex();

    output.position = math::linalg::promoted_mul(pc.spc.instance.transform, vertex);
    output.color = pc.spc.instance.color;

    return output;
}

[shader("vertex")]
PSInput aabb_vertex_instances()
{
    PSInput output;
    const float32_t3 vertex = getUnitAABBVertex();
    InstanceData instance = vk::BufferPointer<InstanceData>(pc.ipc.pInstanceBuffer + sizeof(InstanceData) * glsl::gl_InstanceIndex()).Get();

    output.position = math::linalg::promoted_mul(instance.transform, vertex);
    output.color = instance.color;

    return output;
}

[shader("pixel")]
float32_t4 aabb_fragment(PSInput input) : SV_TARGET
{
    float32_t4 outColor = input.color;

    return outColor;
}
