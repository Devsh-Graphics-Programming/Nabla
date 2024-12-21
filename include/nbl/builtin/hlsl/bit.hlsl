#ifndef _NBL_BUILTIN_HLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BIT_INCLUDED_


#include <nbl/builtin/hlsl/macros.h>


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

#if 0 // enable this if you run into bit_cast not working for a non fundamental type
template<class T, class U>
enable_if_t<sizeof(T)==sizeof(U)&&(is_scalar_v<T>||is_vector_v<T>)==(is_scalar_v<U>||is_vector_v<U>),T> bit_cast(U val)
{
    return spirv::bitcast<T,U>(val);
}
// unfortunately its impossible to deduce Storage Class right now,
// also this function will only work as long as `inout` behaves as `__restrict &` in DXC
template<class T, class U, uint32_t StorageClass>
enable_if_t<sizeof(T)==sizeof(U),T> bit_cast(inout U val)
{
    using ptr_u_t = spirv::pointer_t<U,StorageClass>;
    // get the address of U
    ptr_u_t ptr_u = spirv::copyObject<StorageClass,U>(val);
    using ptr_t_t = spirv::pointer_t<T,StorageClass>;
    // reinterpret cast the pointers
    ptr_t_t ptr_t = spirv::bitcast<ptr_t_t.ptr_u_t>(ptr_u);
    // actually load and return the value
    return spirv::load<T,ptr_t_t>(ptr_t);
}
#else
template<class T, class U>
enable_if_t<sizeof(T)==sizeof(U),T> bit_cast(U val)
{
    return spirv::bitcast<T,U>(val);
}
#endif

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

}
}
#endif
 
#endif
