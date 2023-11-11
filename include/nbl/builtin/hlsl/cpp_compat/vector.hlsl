#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_VECTOR_INCLUDED_

#ifndef __HLSL_VERSION 

#define GLM_FORCE_SWIZZLE
#include <glm/glm/glm.hpp>
#include <glm/glm/detail/_swizzle.hpp>
#include <stdint.h>
#include <half.h>

namespace nbl::hlsl
{

template<typename T, uint16_t N>
using vector = glm::vec<N, T>;

// ideally we should have sized bools, but no idea what they'd be
using bool4 = vector<bool, 4>;
using bool3 = vector<bool, 3>;
using bool2 = vector<bool, 2>;
using bool1 = vector<bool, 1>;

using int32_t4 = vector<int32_t, 4>;
using int32_t3 = vector<int32_t, 3>;
using int32_t2 = vector<int32_t, 2>;
using int32_t1 = vector<int32_t, 1>;

using uint32_t4 = vector<uint32_t, 4>;
using uint32_t3 = vector<uint32_t, 3>;
using uint32_t2 = vector<uint32_t, 2>;
using uint32_t1 = vector<uint32_t, 1>;

// TODO: halfN -> needs class implementation or C++23 std:float16_t

using float16_t = half;
using float16_t4 = vector<float16_t, 4>;
using float16_t3 = vector<float16_t, 3>;
using float16_t2 = vector<float16_t, 2>;
using float16_t1 = vector<float16_t, 1>;

using float32_t = float;
using float32_t4 = vector<float32_t, 4>;
using float32_t3 = vector<float32_t, 3>;
using float32_t2 = vector<float32_t, 2>;
using float32_t1 = vector<float32_t, 1>;

using float64_t = double;
using float64_t4 = vector<float64_t, 4>;
using float64_t3 = vector<float64_t, 3>;
using float64_t2 = vector<float64_t, 2>;
using float64_t1 = vector<float64_t, 1>;

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

#endif
