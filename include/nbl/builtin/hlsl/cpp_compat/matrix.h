#ifndef _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_INCLUDED_

#ifndef __HLSL_VERSION 
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include "glm/detail/_swizzle.hpp"
#include <stdint.h>
#endif

#include <nbl/builtin/hlsl/cpp_compat/vector.h>

namespace nbl::hlsl
{

#ifndef __HLSL_VERSION 
template<typename T, uint16_t N, uint16_t M>
struct matrix final : private glm::mat<N,M,T>
{
    using Base = glm::mat<N,M,T>;
    using Base::Base;
    using Base::operator=;
    using Base::operator[];

    template<uint16_t X, uint16_t Y> matrix(matrix<T, X, Y> const&) = delete;

    matrix(matrix const&) = default;
    explicit matrix(Base const& base) : Base(base) {}

    friend matrix operator*(matrix const& lhs, matrix const& rhs)
    { 
        matrix re = lhs;
        for(int i = 0; i < N; ++i)
            lhs[i] *= rhs[i];
        return re;
    }

    friend matrix operator+(matrix const& lhs, matrix const& rhs){ return matrix(lhs.asBase() + rhs.asBase()); }
    friend matrix operator-(matrix const& lhs, matrix const& rhs){ return matrix(lhs.asBase() - rhs.asBase()); }
    

    template<typename T, uint16_t X, uint16_t Y, uint16_t Z>
    friend matrix<T, X, Z> mul(matrix<T, X, Y> const& lhs, matrix<T, Y, Z> const& rhs);
    template<typename T, uint16_t X, uint16_t Y>
    friend vector<T, X> mul(matrix<T, X, Y> const& lhs, vector<T, Y> const& rhs);
    template<typename T, uint16_t X, uint16_t Y>
    friend vector<T, Y> mul(vector<T, X> const& lhs, matrix<T, X, Y> const& rhs);
private:
    Base const& asBase() const { return *reinterpret_cast<const Base*>(this); }
};

template<typename T, uint16_t X, uint16_t Y, uint16_t Z>
inline matrix<T, X, Z> mul(matrix<T, X, Y> const& lhs, matrix<T, Y, Z> const& rhs) { return matrix<T, X, Z>(rhs.asBase() * lhs.asBase()); }

template<typename T, uint16_t X, uint16_t Y>
inline vector<T, X> mul(matrix<T, X, Y> const& lhs, vector<T, Y> const& rhs) { return rhs * lhs.asBase(); }

template<typename T, uint16_t X, uint16_t Y>
inline vector<T, Y> mul(vector<T, X> const& lhs, matrix<T, X, Y> const& rhs) { return rhs.asBase() * lhs; }

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