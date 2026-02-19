#ifndef _NBL_HLSL_EXT_ENVMAP_IMPORTANCE_SAMPLING_PARAMETERS_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_ENVMAP_IMPORTANCE_SAMPLING_PARAMETERS_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace envmap_importance_sampling
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
