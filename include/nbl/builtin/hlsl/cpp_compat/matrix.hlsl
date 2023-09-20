#ifndef _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_

#ifndef __HLSL_VERSION 
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include "glm/detail/_swizzle.hpp"
#include <stdint.h>
#endif

#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

namespace nbl::hlsl
{

#ifndef __HLSL_VERSION 
template<typename T, uint16_t N, uint16_t M>
struct matrix final : private glm::mat<N,M,T>
{
    using Base = glm::mat<N,M,T>;
    using Base::Base;
    using Base::operator[];

    template<uint16_t X, uint16_t Y, std::enable_if<!(X == N && Y == M) && X <= N && Y <= M>>
    explicit matrix(matrix<T, X, Y> const& m) : Base(m)
    {
    }

    matrix(matrix const&) = default;
    explicit matrix(Base const& base) : Base(base) {}

    matrix& operator=(matrix const& rhs)
    {
        Base::operator=(rhs);
        return *this;
    }

    friend matrix operator+(matrix const& lhs, matrix const& rhs){ return matrix(reinterpret_cast<Base const&>(lhs) + reinterpret_cast<Base const&>(rhs)); }
    friend matrix operator-(matrix const& lhs, matrix const& rhs){ return matrix(reinterpret_cast<Base const&>(lhs) - reinterpret_cast<Base const&>(rhs)); }
    
    template<uint16_t K>
    inline friend matrix<T, N, K> mul(matrix const& lhs, matrix<T, M, K> const& rhs)
    {
        return matrix<T, N, K>(glm::operator*(reinterpret_cast<Base const&>(rhs), reinterpret_cast<matrix<T, M, K>::Base const&>(lhs)));
    }
    
    inline friend vector<T, N> mul(matrix const& lhs, vector<T, M> const& rhs)
    {
        return glm::operator* (rhs, reinterpret_cast<Base const&>(lhs));
    }

    inline friend vector<T, M> mul(vector<T, N> const& lhs, matrix const& rhs)
    {
        return glm::operator*(reinterpret_cast<Base const&>(rhs), lhs);
    }
};

using float4x4 = matrix<float, 4, 4>;
using float4x3 = matrix<float, 4, 3>;
using float4x2 = matrix<float, 4, 2>;
using float3x4 = matrix<float, 3, 4>;
using float3x3 = matrix<float, 3, 3>;
using float3x2 = matrix<float, 3, 2>;
using float2x4 = matrix<float, 2, 4>;
using float2x3 = matrix<float, 2, 3>;
using float2x2 = matrix<float, 2, 2>;

#endif

}

#endif