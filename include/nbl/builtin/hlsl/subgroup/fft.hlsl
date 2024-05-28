#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/complex.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
    // Considering SubgroupSize as a power of 2, no bigger than 128
    template<typename Scalar, uint32_t SubgroupSize, bool inverse>
    complex_t<Scalar> getSubgroupTwiddle(uint32_t twiddleIdx){
                                                     
    }   

}
}
}

#endif