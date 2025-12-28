// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/ieee754.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{
template<typename T, uint32_t LP, bool Odd=(LP&0x1) NBL_STRUCT_CONSTRAINABLE>
struct lp_norm;

// infinity case
template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>)
struct lp_norm<T,0,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>) >
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __call(const T v)
    {
        array_get<T, scalar_type> getter;

        scalar_type retval = abs<scalar_type>(getter(v, 0));
        for (int i = 1; i < vector_traits<T>::Dimension; i++)
            retval = max<scalar_type>(abs<scalar_type>(getter(v, i)),retval);
        return retval;
    }
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>)
struct lp_norm<T,1,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>) >
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __sum(const T v)
    {
        array_get<T, scalar_type> getter;

        scalar_type retval = abs<scalar_type>(getter(v, 0));
        for (int i = 1; i < vector_traits<T>::Dimension; i++)
            retval += abs<scalar_type>(getter(v, i));
        return retval;
    }

    static scalar_type __call(const T v)
    {
        return __sum(v);
    }
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct lp_norm<T,2,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) >
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __sum(const T v)
    {
        return hlsl::dot<T>(v, v);
    }

    static scalar_type __call(const T v)
    {
        return hlsl::sqrt<scalar_type>(__sum(v));
    }
};
}

template<typename T, uint32_t LP NBL_FUNC_REQUIRES((concepts::FloatingPointVector<T> || concepts::FloatingPointVectorial<T>) && LP>0)
scalar_type_t<T> lpNormPreroot(NBL_CONST_REF_ARG(T) v)
{
    return impl::lp_norm<T,LP>::__sum(v);
}

template<typename T, uint32_t LP NBL_FUNC_REQUIRES(concepts::FloatingPointVector<T> || concepts::FloatingPointVectorial<T>)
scalar_type_t<T> lpNorm(NBL_CONST_REF_ARG(T) v)
{
    return impl::lp_norm<T,LP>::__call(v);
}


// valid only for `theta` in [-PI,PI]
template <typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<T>)
void sincos(T theta, NBL_REF_ARG(T) s, NBL_REF_ARG(T) c)
{
    c = cos<T>(theta);
    s = sqrt<T>(T(NBL_FP64_LITERAL(1.0))-c*c);
    s = ieee754::flipSign(s, theta < T(NBL_FP64_LITERAL(0.0)));
}

template <typename T NBL_FUNC_REQUIRES(vector_traits<T>::Dimension == 3)
void frisvad(NBL_CONST_REF_ARG(T) normal, NBL_REF_ARG(T) tangent, NBL_REF_ARG(T) bitangent)
{
    using scalar_t = typename vector_traits<T>::scalar_type;
    const scalar_t unit = _static_cast<scalar_t>(1);

	const scalar_t a = unit / (unit + normal.z);
	const scalar_t b = -normal.x * normal.y * a;
    if (normal.z < -_static_cast<scalar_t>(0.9999999))
    {
        tangent = T(0.0,-1.0,0.0);
        bitangent = T(-1.0,0.0,0.0);
    }
    else
    {
        tangent = T(unit - normal.x * normal.x * a, b, -normal.x);
        bitangent = T(b, unit - normal.y * normal.y * a, -normal.y);
    }
}

namespace impl
{
template <typename T NBL_STRUCT_CONSTRAINABLE>
struct conditionalAbsOrMax_helper;

// TODO: conditionalAbsOrMax_helper partial template specialization for signed integers

template <typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<T>)
struct conditionalAbsOrMax_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<T>) >
{
    static T __call(bool cond, NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) limit)
    {
        using UintOfTSize = unsigned_integer_of_size_t<sizeof(T)>;
        const T condAbs = bit_cast<T>(bit_cast<UintOfTSize>(x) & (cond ? (numeric_limits<UintOfTSize>::max >> 1) : numeric_limits<UintOfTSize>::max));

        return max<T>(condAbs, limit);
    }
};

template <typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct conditionalAbsOrMax_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) >
{
    static T __call(bool cond, NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) limit)
    {
        using UintOfTSize = unsigned_integer_of_size_t<sizeof(typename vector_traits<T>::scalar_type)>;
        const int dimensionOfT = vector_traits<T>::Dimension;
        using Uint32VectorWithDimensionOfT = vector<uint32_t, dimensionOfT>;
        using scalar_type = typename vector_traits<T>::scalar_type;

        Uint32VectorWithDimensionOfT xAsUintVec = bit_cast<Uint32VectorWithDimensionOfT, T>(x);

        const Uint32VectorWithDimensionOfT mask = cond ? _static_cast<Uint32VectorWithDimensionOfT>(numeric_limits<UintOfTSize>::max >> 1) : _static_cast<Uint32VectorWithDimensionOfT>(numeric_limits<UintOfTSize>::max);
        const Uint32VectorWithDimensionOfT condAbsAsUint = xAsUintVec & mask;
        T condAbs = bit_cast<T, Uint32VectorWithDimensionOfT>(condAbsAsUint);

        return max<T>(condAbs, limit);
    }
};
}

// @ return abs(x) if cond==true, max(x,0.0) otherwise
template <typename T>
T conditionalAbsOrMax(bool cond, T x, T limit)
{
    return impl::conditionalAbsOrMax_helper<T>::__call(cond, x, limit);
}

template<typename Lhs, typename Rhs NBL_FUNC_REQUIRES(concepts::Matricial<Lhs> && concepts::Matricial<Rhs> && (matrix_traits<Lhs>::ColumnCount == matrix_traits<Rhs>::RowCount))
typename cpp_compat_intrinsics_impl::mul_helper<Lhs, Rhs>::return_t applyChainRule(Lhs dFdG, Rhs dGdR)
{
    return hlsl::mul(dFdG, dGdR);
}

}
}
}

#endif
