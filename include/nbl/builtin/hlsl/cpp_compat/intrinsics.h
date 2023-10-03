#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>


// this is a C++ only header, hence the `.h` extension, it only implements HLSL's built-in functions
#ifndef __HLSL_VERSION
namespace nbl::hlsl
{


template<typename T>
inline T cross(const vector<T,3>& lhs, const vector<T,3>& rhs)
{
    return glm::cross(lhs,rhs);
}

template<typename T, typename U>
inline T lerp(const T& lhs, const T& rhs, const U& t)
{
    return glm::mix(lhs,rhs,t);
}


}
#endif

#endif
