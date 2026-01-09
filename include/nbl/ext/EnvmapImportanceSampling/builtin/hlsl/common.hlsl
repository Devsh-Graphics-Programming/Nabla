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
  float32_t4 luminanceScales;
  uint32_t2 lumaMapResolution;
};

struct SLumaMeasurePushConstants
{
  float32_t4 luminanceScales;
  uint32_t2 lumaMapResolution;
  uint64_t lumaMeasurementBuf;
};

struct SLumaMeasurement 
{
  float32_t3 weightedDir;
  float32_t luma;
  float32_t maxLuma;
};

struct device_capabilities
{
#ifdef TEST_NATIVE
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = true;
#else
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = false;
#endif
};

}
}
}
}

#endif
