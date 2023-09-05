#ifndef _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_

#ifdef __cplusplus
#include <glm/glm.hpp>
#include <stdint.h>
#endif

namespace nbl::hlsl
{

#ifdef __cplusplus
template<typename T, uint16_t N, uint16_t M>
struct matrix final : private glm::mat<N,M,T>
{
    using Base = glm::mat<N,M,T>;
    using Base::Base;
    using Base::operator=;

    template<uint16_t X, uint16_t Y> matrix(matrix<T, X, Y> const&) = delete;

    matrix(matrix const&) = default;
    matrix(Base const& base) : Base(base) {}

    Base const& asBase() const { return *reinterpret_cast<const Base*>(this); }

    
    template<class T, uint16_t K>
    friend matrix<T, N, K> operator*(matrix const& lhs, matrix<T, M, K> const& rhs){ return rhs.asBase() * lhs.asBase(); }

    friend matrix operator+(matrix const& lhs, matrix const& rhs){ return matrix(lhs.asBase() + rhs.asBase()); }
    friend matrix operator-(matrix const& lhs, matrix const& rhs){ return matrix(lhs.asBase() - rhs.asBase()); }
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

template<class T, uint16_t X, uint16_t Y, uint16_t Z>
inline matrix<T, X, Z> mul(matrix<T, X, Y> const& lhs, matrix<T, Y, Z> const& rhs){ return rhs.asBase() * lhs.asBase(); }
#endif

}

#endif