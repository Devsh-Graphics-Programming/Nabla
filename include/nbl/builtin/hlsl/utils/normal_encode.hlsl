
#ifndef _NBL_BUILTIN_HLSL_UTILS_NORMAL_ENCODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_NORMAL_ENCODE_INCLUDED_

#include "nbl/builtin/hlsl/math/constants.hlsl"
#include "nbl/builtin/hlsl/utils/compressed_normal_matrix_t.hlsl"


namespace nbl
{
namespace hlsl
{
namespace normal_encode
{


float2 signedSpherical(in float3 n)
{
    return float2(atan2(n.y,n.x)/math::PI, n.z);
}


}
}
}

#endif