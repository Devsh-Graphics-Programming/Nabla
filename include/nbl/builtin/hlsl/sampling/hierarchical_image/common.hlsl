#ifndef _NBL_HLSL_SAMPLING_HIERARCHICAL_IMAGE_COMMON_INCLUDED_
#define _NBL_HLSL_SAMPLING_HIERARCHICAL_IMAGE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace hierarchical_image
{

struct SLumaGenPushConstants
{
  float32_t3 lumaRGBCoefficients;
  uint32_t lumaMapWidth : 16;
  uint32_t lumaMapHeight : 16;
  uint16_t lumaMapLayer;
};

struct SWarpGenPushConstants
{
  uint32_t lumaMapWidth : 16;
  uint32_t lumaMapHeight : 16;
  uint32_t warpMapWidth : 16;
  uint32_t warpMapHeight : 16;
  // Both warpMap and lumaMap should have the same layer count
  uint16_t lumaMapLayer;
};

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t GenWarpWorkgroupDim = 16;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t GenLumaWorkgroupDim = 16;

}
}
}
}

#endif
