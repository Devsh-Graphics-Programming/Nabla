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
#ifdef __HLSL_VERSION
    float32_t3x4 transform;
#else
    float transform[3*4];
#endif
    nbl::hlsl::float32_t4 color;
};

struct SPushConstants
{
#ifdef __HLSL_VERSION
    float32_t4x4 MVP;
#else
    float MVP[4*4];
#endif
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
