#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_


#ifndef __HLSL_VERSION
#include <type_traits>
#include <bit>

#define ARROW ->
#define NBL_CONSTEXPR constexpr
#define NBL_CONSTEXPR_STATIC constexpr static
#define NBL_CONSTEXPR_STATIC_INLINE constexpr static inline

namespace nbl::hlsl
{

template<class T, class U>
constexpr T bit_cast(const U& val)
{
    return std::bit_cast<T, U>(val);
}

template<typename T>
using add_reference = std::add_lvalue_reference<T>;

template<typename T>
using add_pointer = std::add_pointer<T>;

}

#define NBL_REF_ARG(T) typename nbl::hlsl::add_reference<T>::type
#define NBL_CONST_REF_ARG(T) typename nbl::hlsl::add_reference<std::add_const_t<T>>::type

// it includes vector and matrix
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.h>

#else

#define ARROW .arrow().
#define NBL_CONSTEXPR const static
#define NBL_CONSTEXPR_STATIC_INLINE const static

namespace nbl
{
namespace hlsl
{

namespace impl
{
    template<class T> float16_t asfloat16_t(T val) { return asfloat16(val); }
    template<class T> float32_t asfloat32_t(T val) { return asfloat(val); }
    template<class T> int16_t   asint16_t(T val) { return asint16(val); }
    template<class T> int32_t   asint32_t(T val) { return asint(val); }
    template<class T> uint16_t  asuint16_t(T val) { return asuint16(val); }
    template<class T> uint32_t  asuint32_t(T val) { return asuint(val); }
    
    template<class T>
    float64_t asfloat64_t(T val) 
    { 
        uint64_t us = uint64_t(val);
        return asdouble(uint32_t(val & ~0u), uint32_t((val >> 32u) & ~0u)); 
    }
    
    template<>
    float64_t asfloat64_t<float64_t>(float64_t val) { return val; }

    template<class T> uint64_t asuint64_t(T val) {  return val; }
    template<class T> int64_t asint64_t(T val) {  return asuint64_t(val); }
    
    template<>
    uint64_t asuint64_t<float64_t>(float64_t val) 
    { 
        uint32_t lo, hi;
        asuint(val, lo, hi);
        return (uint64_t(hi) << 32u) | uint64_t(lo);
    }
}

template<class T, class U = T>
T bit_cast(U val);

#define NBL_DECLARE_BIT_CAST(FROM, TO) template<> TO bit_cast<TO, FROM>(FROM val) { return impl::as##TO (val); }

#define NBL_DECLARE_BIT_CAST_TYPES(BASE, BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, float ## BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, uint ## BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, int ## BITS)

#define NBL_DECLARE_BIT_CAST_BITS(BASE) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 16_t) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 32_t) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 64_t)

NBL_DECLARE_BIT_CAST_BITS(float)
NBL_DECLARE_BIT_CAST_BITS(uint)
NBL_DECLARE_BIT_CAST_BITS(int)

#undef NBL_DECLARE_BIT_CAST
#undef NBL_DECLARE_BIT_CAST_TYPES
#undef NBL_DECLARE_BIT_CAST_BITS

#if 0 // TODO: for later
template<typename T>
struct add_reference
{
  using type = ref<T>;
};
template<typename T>
struct add_pointer
{
  using type = ptr<T>;
};
#endif

}
}

#define NBL_REF_ARG(T) inout T
#define NBL_CONST_REF_ARG(T) const in T

#endif

#endif