#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

// this is a C++ only header, hence the `.h` extension, it only implements HLSL's built-in functions
#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>
#include "nbl/core/util/bitflag.h"

namespace nbl::hlsl
{
#define NBL_SIMPLE_GLM_PASSTHROUGH(HLSL_ID,GLSL_ID,...) template<typename... Args>\
inline auto HLSL_ID(Args&&... args) \
{ \
    return glm::GLSL_ID(std::forward<Args>(args)...);\
}
#define NBL_BIT_OP_GLM_PASSTHROUGH(HLSL_ID,GLSL_ID) template<typename T> \
inline auto HLSL_ID(const T bitpattern) \
{ \
    if constexpr (std::is_integral_v<T>) \
        return glm::GLSL_ID(bitpattern); \
    else \
    { \
        if constexpr (std::is_enum_v<T>) \
        { \
            const auto as_underlying = static_cast<std::underlying_type_t<T>>(bitpattern); \
            return glm::GLSL_ID(as_underlying); \
        } \
        else \
        { \
            if constexpr (std::is_same_v<T,core::bitflag<typename T::enum_t>>) \
                return HLSL_ID<typename T::enum_t>(bitpattern.value); \
        } \
    } \
}

NBL_BIT_OP_GLM_PASSTHROUGH(bitCount,bitCount)

NBL_SIMPLE_GLM_PASSTHROUGH(cross,cross)
NBL_SIMPLE_GLM_PASSTHROUGH(clamp,clamp)

template<typename T>
inline typename scalar_type<T>::type dot(const T& lhs, const T& rhs) {return glm::dot(lhs,rhs);}

// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename T, uint16_t N, uint16_t M>
inline T determinant(const matrix<T,N,M>& m)
{
    return glm::determinant(reinterpret_cast<typename matrix<T,N,M>::Base const&>(m));
}

NBL_BIT_OP_GLM_PASSTHROUGH(findLSB,findLSB)

NBL_BIT_OP_GLM_PASSTHROUGH(findMSB,findMSB)

// inverse not defined cause its implemented via hidden friend
template<typename T, uint16_t N, uint16_t M>
inline matrix<T,N,M> inverse(const matrix<T,N,M>& m)
{
    return glm::inverse(reinterpret_cast<typename matrix<T,N,M>::Base const&>(m));
}

NBL_SIMPLE_GLM_PASSTHROUGH(lerp,mix)

// transpose not defined cause its implemented via hidden friend
template<typename T, uint16_t N, uint16_t M>
inline matrix<T,M,N> transpose(const matrix<T,N,M>& m)
{
    return glm::transpose(reinterpret_cast<typename matrix<T,N,M>::Base const&>(m));
}

template<typename T>
inline T bitfieldExtract( T val, uint32_t offsetBits, uint32_t numBits )
{
    return glm::bitfieldExtract( val, int32_t( offsetBits ), int32_t( numBits ) );
}

#undef NBL_BIT_OP_GLM_PASSTHROUGH
#undef NBL_SIMPLE_GLM_PASSTHROUGH

#define NBL_ALIAS_TEMPLATE_FUNCTION(origFunctionName, functionAlias) \
template<typename... Args> \
inline auto functionAlias(Args&&... args) -> decltype(origFunctionName(std::forward<Args>(args)...)) \
{ \
    return origFunctionName(std::forward<Args>(args)...); \
}

NBL_ALIAS_TEMPLATE_FUNCTION(std::min, min);
NBL_ALIAS_TEMPLATE_FUNCTION(std::max, max);
NBL_ALIAS_TEMPLATE_FUNCTION(std::isnan, isnan);
NBL_ALIAS_TEMPLATE_FUNCTION(std::isinf, isinf);
NBL_ALIAS_TEMPLATE_FUNCTION(std::exp2, exp2);

template<typename T>
inline T rsqrt(T x)
{
    return 1.0f / std::sqrt(x);
}


}
#endif

#endif
