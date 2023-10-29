#ifndef _NBL_BUILTIN_HLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BIT_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>

#ifndef __HLSL_VERSION
#include <bit>

namespace nbl::hlsl
{

NBL_ALIAS_TEMPLATE_FUNCTION(std::rotl, rotl);
NBL_ALIAS_TEMPLATE_FUNCTION(std::rotr, rotr);
NBL_ALIAS_TEMPLATE_FUNCTION(std::countl_zero, countl_zero);
NBL_ALIAS_TEMPLATE_FUNCTION(std::bit_cast, bit_cast);

}
#else
namespace nbl
{
namespace hlsl
{

template<class T, class U>
T bit_cast(U val)
{
    static_assert(sizeof(T) <= sizeof(U), "destination type must be less than or equal to the source type in size");
    return spirv::bitcast<T, U>(val);
}

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

namespace impl
{
template<uint16_t bits>
uint16_t clz(uint64_t N)
{
    static const uint64_t SHIFT   = bits>>1;
    static const uint64_t LO_MASK = (1ull<<SHIFT)-1;
    const bool CHOOSE_HIGH = N & (LO_MASK<<SHIFT);
    const uint64_t NEXT = (CHOOSE_HIGH ? (N>>SHIFT):N)&LO_MASK;
    const uint16_t value = uint16_t(clz<SHIFT>(NEXT) + (CHOOSE_HIGH ? 0:SHIFT));
    return value;
}


template<>
uint16_t clz<1>(uint64_t N) { return uint16_t(1u-N&1); }

}

template<typename T>
uint16_t countl_zero(T n)
{
    return impl::clz<sizeof(T)*8>(n);
}

}
}
#endif
 
#endif
