#pragma shader_stage(vertex)

#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::debug_draw;

[[vk::push_constant]] SPushConstants pc;

[shader("vertex")]
PSInput main()
{
    const float32_t3 unitAABBVertices[8] = {
        float32_t3(0.0, 0.0, 0.0),
        float32_t3(1.0, 0.0, 0.0),
        float32_t3(0.0, 0.0, 1.0),
        float32_t3(1.0, 0.0, 1.0),
        float32_t3(0.0, 1.0, 0.0),
        float32_t3(1.0, 1.0, 0.0),
        float32_t3(0.0, 1.0, 1.0),
        float32_t3(1.0, 1.0, 1.0)
    };

    PSInput output;
    float32_t3 vertex = unitAABBVertices[glsl::gl_VertexIndex()];
    InstanceData instance = vk::BufferPointer<InstanceData>(pc.pInstanceBuffer + sizeof(InstanceData) * glsl::gl_InstanceIndex()).Get();

    output.position = math::linalg::promoted_mul(instance.transform, vertex);
    output.color = instance.color;

    return output;
}