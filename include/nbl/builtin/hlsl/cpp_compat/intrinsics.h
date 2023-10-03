#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>


// this is a C++ only header, hence the `.h` extension, it only implements HLSL's built-in functions
#ifndef __HLSL_VERSION
namespace nbl::hlsl
{
#define NBL_SIMPLE_GLM_PASSTHROUGH(HLSL_ID,GLSL_ID,...) template<typename... Args>\
inline auto HLSL_ID(Args&&... args) \
{ \
    return glm::GLSL_ID(std::forward<Args>(args)...);\
}


NBL_SIMPLE_GLM_PASSTHROUGH(cross,cross)

template<typename T>
inline T dot(const T& lhs, const T& rhs) {return glm::dot(lhs,rhs);}

// inverse not listed cause it needs friendship

NBL_SIMPLE_GLM_PASSTHROUGH(lerp,mix)

// transpose not listed cause it needs friendship

#undef NBL_SIMPLE_GLM_PASSTHROUGH
}
#endif

#endif
