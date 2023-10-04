#ifndef _NBL_BUILTIN_HLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BIT_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>


#ifndef __HLSL_VERSION
#include <bit>
#else
namespace nbl
{
namespace hlsl
{

uint32_t rotl(NBL_CONST_REF_ARG(uint32_t) x, NBL_CONST_REF_ARG(uint32_t) s)
{
    const uint32_t N = 32u;
    const uint32_t r = s % N;
    
    if(r == 0)
        return 0;
    
    if(r > 0)
    {
        return (x << r) | (x >> (N - r));
    }
    else
    {
        return std::rotr(x, -r);
    }
}

uint32_t rotr(NBL_CONST_REF_ARG(uint32_t) x, NBL_CONST_REF_ARG(uint32_t) s)
{
    const uint32_t N = 32u;
    const uint32_t r = s % N;
    
    if(r == 0)
        return 0;
    
    if(r > 0)
    {
        return (x >> r) | (x << (N - r));
    }
    else
    {
        return std::rotl(x, -r);
    }
}

}
}
#endif
 
#endif