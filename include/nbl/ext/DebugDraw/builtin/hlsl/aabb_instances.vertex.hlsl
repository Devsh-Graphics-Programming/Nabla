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
    PSInput output;

    float32_t3 vertex = (bda::__ptr<float32_t3>::create(pc.pVertexBuffer) + glsl::gl_VertexIndex()).deref_restrict().load();
    InstanceData instance = vk::BufferPointer<InstanceData>(pc.pInstanceBuffer + sizeof(InstanceData) * glsl::gl_InstanceIndex()).Get();

    float32_t4x4 transform;
    transform[0] = instance.transform[0];
    transform[1] = instance.transform[1];
    transform[2] = instance.transform[2];
    transform[3] = float32_t4(0, 0, 0, 1);
    float32_t4 position = math::linalg::promoted_mul(transform, vertex);
    output.position = mul(pc.MVP, position);
    output.color = instance.color;

    return output;
}