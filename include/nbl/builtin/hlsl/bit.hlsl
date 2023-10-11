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

}
}
#endif
 
#endif