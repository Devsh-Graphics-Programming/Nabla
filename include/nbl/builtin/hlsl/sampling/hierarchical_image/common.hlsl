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
  uint32_t2 lumaMapResolution;
};

}
}
}
}

#endif
