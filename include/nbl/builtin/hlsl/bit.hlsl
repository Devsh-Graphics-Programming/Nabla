#ifndef _NBL_BUILTIN_HLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BIT_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

#ifndef __HLSL_VERSION
#include <bit>

namespace nbl::hlsl
{

NBL_ALIAS_TEMPLATE_FUNCTION(std::rotl, rotl);
NBL_ALIAS_TEMPLATE_FUNCTION(std::rotr, rotr);

}
#else
namespace nbl
{
namespace hlsl
{

template<typename T, typename S>
T rotl(T x, S s);
template<typename T, typename S>
T rotr(T x, S s);

template<typename T, typename S>
T rotl(T x, S s)
{
    const T N = 32u;
    const S r = s % N;
    
    if(r >= 0)
    {
        return (x << r) | (x >> (N - r));
    }
    else
    {
        return (x >> (-r)) | (x << (N - (-r)));
    }
}

template<typename T, typename S>
T rotr(T x, S s)
{
    const T N = 32u;
    const S r = s % N;
    
    if(r >= 0)
    {
        return (x >> r) | (x << (N - r));
    }
    else
    {
        return (x << (-r)) | (x >> (N - (-r)));
    }
}

template<typename T>
uint16_t countl_zero(T n)
{
    uint16_t result = 0u;
    for(uint32_t bits_log2 = 6u; bits_log2 >= 0u; bits_log2--)
    {
        const uint16_t shift = bits_log2 ? uint16_t(1)<<(bits_log2-1) : 0;
        const uint64_t loMask = bits_log2 ? (1ull<<shift)-1 : 0;
        const bool chooseHigh = n&(loMask<<shift);
        n = T(chooseHigh ? (n>shift):n)&loMask;

        result += uint16_t(chooseHigh ? 0ull : shift);
    }

    return result;
}

}
}
#endif
 
#endif