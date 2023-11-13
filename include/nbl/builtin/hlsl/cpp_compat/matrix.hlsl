#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_MATRIX_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_MATRIX_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

namespace nbl
{
namespace hlsl
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

    inline friend matrix inverse(matrix const& m) 
    {
        return matrix(glm::inverse(reinterpret_cast<Base const&>(m)));
    }
    
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

    inline friend matrix transpose(matrix const& m)
    {
        return glm::transpose(reinterpret_cast<Base const&>(m));
    }
};
#endif


#define NBL_TYPEDEF_MATRICES_FOR_ROW(T, R) \
typedef matrix<T, R, 4> T ## R ## x4; \
typedef matrix<T, R, 3> T ## R ## x3; \
typedef matrix<T, R, 2> T ## R ## x2; 

#define NBL_TYPEDEF_MATRICES_FOR_SCALAR(T) \
    NBL_TYPEDEF_MATRICES_FOR_ROW(T, 4) \
    NBL_TYPEDEF_MATRICES_FOR_ROW(T, 3) \
    NBL_TYPEDEF_MATRICES_FOR_ROW(T, 2)

NBL_TYPEDEF_MATRICES_FOR_SCALAR(bool);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(int16_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(int32_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(int64_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(uint16_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(uint32_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(uint64_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(float16_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(float32_t);
NBL_TYPEDEF_MATRICES_FOR_SCALAR(float64_t);

#undef NBL_TYPEDEF_MATRICES_FOR_ROW
#undef NBL_TYPEDEF_MATRICES_FOR_SCALAR
}
}

#endif
