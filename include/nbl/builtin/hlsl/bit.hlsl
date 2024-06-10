#ifndef _NBL_BUILTIN_HLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BIT_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>


#ifndef __HLSL_VERSION

#include <bit>

namespace nbl::hlsl
{

// NBL_ALIAS_TEMPLATE_FUNCTION variadic templates aren't always appropriate
template<typename To, typename From>
constexpr To bit_cast(const From& from)
{
    return std::bit_cast<To,From>(from);
}

NBL_ALIAS_TEMPLATE_FUNCTION(std::rotl, rotl);
NBL_ALIAS_TEMPLATE_FUNCTION(std::rotr, rotr);
NBL_ALIAS_TEMPLATE_FUNCTION(std::countl_zero, countl_zero);

}
#else

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>

namespace nbl
{
namespace hlsl
{

template<class T, class U>
T bit_cast(U val)
{
    static_assert(sizeof(T) <= sizeof(U));
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

} //namespace impl

template<typename T>
uint16_t countl_zero(T n)
{
    return impl::clz<sizeof(T)*8>(n);
}


namespace impl {
uint32_t bitfieldInsert(uint32_t base, uint32_t shifted_masked_value, uint32_t lo, uint32_t count)
{
    const uint32_t hi = base^lo;
    return (hi<<count)|shifted_masked_value|lo;
}

} //namespace impl

uint32_t bitfieldInsert(uint32_t base, uint32_t value, uint32_t offset, uint32_t count)
{
    const uint32_t shifted_masked_value = (value&((1u<<count)-1))<<offset;
    return impl::bitfieldInsert(base,shifted_masked_value,base&((1u<<offset)-1),count);
}

}
}
#endif
 
#endif
