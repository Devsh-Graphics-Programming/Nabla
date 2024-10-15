#ifndef _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace central_limit_blur
{

struct BoxBlurParams
{
  uint32_t flip;
  uint32_t filterDim;
  uint32_t blockDim;
};

}
}
}
#endif