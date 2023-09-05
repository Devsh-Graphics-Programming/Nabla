
#ifndef _NBL_BUILTIN_HLSL_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.h>

namespace nbl::hlsl
{

#ifdef __cplusplus
template<typename T, uint16_t N>
struct vector final : private glm::vec<N,T>
{
    using Base = glm::vec<N,T>;
    using Base::Base;
    using Base::operator=;

    template<uint16_t X> vector(vector<T,X> const&) = delete;

    vector(vector const&) = default;
    vector(Base const& base) : Base(base) {}

    Base const& asBase() const { return *reinterpret_cast<const Base*>(this); }

    template<class T, uint16_t R>
    friend vector<T, R> operator*(matrix<T, R, N> const& lhs, vector const& rhs)
    { 
        return rhs.asBase() * lhs.asBase(); 
    }

    template<class T, uint16_t C>
    friend vector<T, C> operator*(vector const& lhs, matrix<T, N, C> const& rhs)
    {
        return rhs.asBase() * lhs.asBase();
    }

    friend vector operator+(vector const& lhs, vector const& rhs){ return lhs.asBase() + rhs.asBase(); }
    friend vector operator-(vector const& lhs, vector const& rhs){ return lhs.asBase() - rhs.asBase(); }
};

using float4 = vector<float, 4>;
using float3 = vector<float, 3>;
using float2 = vector<float, 2>;

#endif

}

#endif