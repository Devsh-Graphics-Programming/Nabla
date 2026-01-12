#ifndef _NBL_DEBUG_DRAW_EXT_COMMON_HLSL
#define _NBL_DEBUG_DRAW_EXT_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#endif

namespace nbl
{
namespace ext
{
namespace debug_draw
{

struct InstanceData
{
    hlsl::float32_t4x4 transform;
    hlsl::float32_t4 color;
};

struct SSinglePC
{
    InstanceData instance;
};

struct SInstancedPC
{
    uint64_t pInstanceBuffer;
};

struct PushConstants
{
    SSinglePC spc;
    SInstancedPC ipc;
};

#ifdef __HLSL_VERSION
struct PSInput
{
    float32_t4 position : SV_Position;
    nointerpolation float32_t4 color : TEXCOORD0;
};

float32_t3 getUnitAABBVertex()
{
    return (hlsl::promote<uint32_t3>(hlsl::glsl::gl_VertexIndex()) >> uint32_t3(0,2,1)) & 0x1u;
}
#endif

}
}
}
#endif
