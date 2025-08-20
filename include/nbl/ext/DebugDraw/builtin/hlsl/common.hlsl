#ifndef _NBL_DEBUG_DRAW_EXT_COMMON_HLSL
#define _NBL_DEBUG_DRAW_EXT_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

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

struct SSinglePushConstants
{
    uint64_t pVertexBuffer;
    InstanceData instance;
};

struct SPushConstants
{
    uint64_t pVertexBuffer;
    uint64_t pInstanceBuffer;
};

#ifdef __HLSL_VERSION
struct PSInput
{
    float32_t4 position : SV_Position;
    float32_t4 color : TEXCOORD0;
};
#endif

}
}
}
#endif
