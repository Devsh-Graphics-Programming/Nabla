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

    // For assigning to same dimension and type use implicit ctor, and even then only allow for dimension truncation
    template<typename U, uint16_t X, uint16_t Y> requires ((!std::is_same_v<T,U> || X!=N || Y!=M) && X>=N && Y>=M)
    explicit matrix(matrix<U,X,Y> const& m) : Base(reinterpret_cast<glm::mat<X,Y,U> const&>(m)) {}

    matrix(matrix const&) = default;
    explicit matrix(Base const& base) : Base(base) {}

    matrix& operator=(matrix const& rhs)
    {
        Base::operator=(rhs);
        return *this;
    }

    // not sure how to forward this
    //inline friend matrix operator*(matrix const& lhs, T rhs) {return matrix(reinterpret_cast<Base const&>(lhs)*rhs);}

    // scalar compound assignment multiply and divide
    inline matrix& operator*=(const T rhs) {return reinterpret_cast<matrix&>(Base::template operator*=(rhs));}
    inline matrix& operator/=(const T rhs) {return reinterpret_cast<matrix&>(Base::template operator/=(rhs));}

    inline friend matrix operator+(matrix const& lhs, matrix const& rhs){ return matrix(reinterpret_cast<Base const&>(lhs) + reinterpret_cast<Base const&>(rhs)); }
    inline friend matrix operator-(matrix const& lhs, matrix const& rhs){ return matrix(reinterpret_cast<Base const&>(lhs) - reinterpret_cast<Base const&>(rhs)); }

    template<uint16_t K>
    inline friend matrix<T, N, K> mul(matrix const& lhs, matrix<T, M, K> const& rhs)
    {
        return matrix<T, N, K>(glm::operator*(reinterpret_cast<matrix<T, M, K>::Base const&>(rhs), reinterpret_cast<Base const&>(lhs)));
    }
    inline friend vector<T, N> mul(matrix const& lhs, vector<T, M> const& rhs)
    {
        return glm::operator* (rhs, reinterpret_cast<Base const&>(lhs));
    }
    inline friend vector<T, M> mul(vector<T, N> const& lhs, matrix const& rhs)
    {
        return glm::operator*(reinterpret_cast<Base const&>(rhs), lhs);
    }
    template<typename ScalarT>
    inline friend matrix mul(const ScalarT lhs, matrix const& rhs)
    {
        return matrix(glm::operator*(lhs, reinterpret_cast<Base const&>(rhs)));
    }
    template<typename ScalarT>
    inline friend matrix mul(matrix const& lhs, const ScalarT rhs)
    {
        return matrix(glm::operator*(reinterpret_cast<Base const&>(lhs), rhs));

    inline friend bool operator==(matrix const& lhs, matrix const& rhs)
    {
        return glm::operator==(reinterpret_cast<Base const&>(lhs), reinterpret_cast<Base const&>(rhs));
    }
    inline friend bool operator!=(matrix const& lhs, matrix const& rhs)
    {
        return glm::operator!=(reinterpret_cast<Base const&>(lhs), reinterpret_cast<Base const&>(rhs));
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

#ifndef __HLSL_VERSION
namespace core
{
template<typename T, uint16_t N, uint16_t M, typename Dummy>
struct blake3_hasher::update_impl<hlsl::matrix<T,N,M>,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const hlsl::matrix<T,N,M>& input)
	{
        hasher.update(&input,sizeof(input));
	}
};
}
#endif
}

#endif
