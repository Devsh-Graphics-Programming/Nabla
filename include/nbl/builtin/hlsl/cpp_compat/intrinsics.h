#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

// this is a C++ only header, hence the `.h` extension, it only implements HLSL's built-in functions
#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>

namespace nbl::hlsl
{
#define NBL_SIMPLE_GLM_PASSTHROUGH(HLSL_ID,GLSL_ID,...) template<typename... Args>\
inline auto HLSL_ID(Args&&... args) \
{ \
    return glm::GLSL_ID(std::forward<Args>(args)...);\
}


NBL_SIMPLE_GLM_PASSTHROUGH(cross,cross)
NBL_SIMPLE_GLM_PASSTHROUGH(clamp,clamp)

template<typename T>
inline typename scalar_type<T>::type dot(const T& lhs, const T& rhs) {return glm::dot(lhs,rhs);}

// inverse not listed cause it needs friendship

NBL_SIMPLE_GLM_PASSTHROUGH(lerp,mix)

// transpose not listed cause it needs friendship

#undef NBL_SIMPLE_GLM_PASSTHROUGH

#define NBL_ALIAS_TEMPLATE_FUNCTION(origFunctionName, functionAlias) \
template<typename... Args> \
inline auto functionAlias(Args&&... args) -> decltype(origFunctionName(std::forward<Args>(args)...)) \
{ \
    return origFunctionName(std::forward<Args>(args)...); \
}

NBL_ALIAS_TEMPLATE_FUNCTION(std::min, min);
NBL_ALIAS_TEMPLATE_FUNCTION(std::max, max);

template<typename T>
inline T rsqrt(T x)
{
    return 1.0f / std::sqrt(x);
}


}
#endif

#endif
