#pragma shader_stage(vertex)

#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "common.hlsl"

using namespace nbl::hlsl;
using namespace nbl::ext::debug_draw;

[[vk::push_constant]] SSinglePushConstants pc;

[shader("vertex")]
PSInput main()
{
    PSInput output;
    float32_t3 vertex = (bda::__ptr<float32_t3>::create(pc.pVertexBuffer) + glsl::gl_VertexIndex()).deref_restrict().load();

    output.position = math::linalg::promoted_mul(pc.instance.transform, vertex);
    output.color = pc.instance.color;

    return output;
}