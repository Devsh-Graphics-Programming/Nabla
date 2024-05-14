#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_VECTOR_INCLUDED_

// stuff for C++
#ifndef __HLSL_VERSION 
#include <stdint.h>

#include <half.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm/glm.hpp>
#include <glm/glm/detail/_swizzle.hpp>

namespace nbl::hlsl
{
template<typename T, uint16_t N>
using vector = glm::vec<N, T>;

template<typename T, uint16_t N>
glm::vec<N, bool> operator<(const glm::vec<N, T>& lhs, const glm::vec<N, T>& rhs)
{
    return glm::lessThan<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, bool> operator>(const glm::vec<N, T>& lhs, const glm::vec<N, T>& rhs)
{
    return glm::greaterThan<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, bool> operator<=(const glm::vec<N, T>& lhs, const glm::vec<N, T>& rhs)
{
    return glm::lessThanEqual<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, bool> operator>=(const glm::vec<N, T>& lhs, const glm::vec<N, T>& rhs)
{
    return glm::greaterThanEqual<N, T>(lhs, rhs);
}

}
#endif

// general typedefs for both langs
namespace nbl
{
namespace hlsl
{
typedef half float16_t;
typedef float float32_t;
typedef double float64_t;

#define NBL_TYPEDEF_VECTORS(T) \
typedef vector<T,4> T ## 4; \
typedef vector<T,3> T ## 3; \
typedef vector<T,2> T ## 2; \
typedef vector<T,1> T ## 1

// ideally we should have sized bools, but no idea what they'd be
NBL_TYPEDEF_VECTORS(bool);

NBL_TYPEDEF_VECTORS(int16_t);
NBL_TYPEDEF_VECTORS(int32_t);
NBL_TYPEDEF_VECTORS(int64_t);

NBL_TYPEDEF_VECTORS(uint16_t);
NBL_TYPEDEF_VECTORS(uint32_t);
NBL_TYPEDEF_VECTORS(uint64_t);

NBL_TYPEDEF_VECTORS(float16_t);
NBL_TYPEDEF_VECTORS(float32_t);
NBL_TYPEDEF_VECTORS(float64_t);

#undef NBL_TYPEDEF_VECTORS
}
}

#endif
