
#ifndef _NBL_BUILTIN_HLSL_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl::hlsl
{

#ifndef __HLSL_VERSION 

template<typename T, uint16_t N>
using vector = glm::vec<N, T>;

using float4 = vector<float, 4>;
using float3 = vector<float, 3>;
using float2 = vector<float, 2>;
using float1 = vector<float, 1>;

using int4 = vector<int32_t, 4>;
using int3 = vector<int32_t, 3>;
using int2 = vector<int32_t, 2>;
using int1 = vector<int32_t, 1>;

using uint4 = vector<uint32_t, 4>;
using uint3 = vector<uint32_t, 3>;
using uint2 = vector<uint32_t, 2>;
using uint1 = vector<uint32_t, 1>;

using bool4 = vector<bool, 4>;
using bool3 = vector<bool, 3>;
using bool2 = vector<bool, 2>;
using bool1 = vector<bool, 1>;

template<typename T, uint16_t N>
glm::vec<N, T> operator<(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return glm::lessThan<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, T> operator>(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return glm::greaterThan<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, T> operator<=(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return glm::lessThanEqual<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, T> operator>=(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return glm::greaterThanEqual<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, T> operator==(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return glm::equal<N, T>(lhs, rhs);
}

template<typename T, uint16_t N>
glm::vec<N, T> operator!=(glm::vec<N, T>& lhs, glm::vec<N, T>& rhs)
{
    return notEqual<N, T>(lhs, rhs);
}

#endif

}

#endif