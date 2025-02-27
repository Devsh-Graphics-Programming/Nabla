// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
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

        scalar_type retval = abs<T>(getter(v, 0));
        for (int i = 1; i < extent<T>::value; i++)
            retval = max<T>(abs<T>(getter(v, i)),retval);
        return retval;
    }
};

// TOOD: is this doing what it should be?
template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>)
struct lp_norm<T,1,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T> || concepts::IntVectorial<T>) >
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __sum(const T v)
    {
        array_get<T, scalar_type> getter;

        scalar_type retval = abs<T>(getter(v, 0));
        for (int i = 1; i < extent<T>::value; i++)
            retval += abs<T>(getter(v, i));
        return retval;
    }

    static scalar_type __call(const T v)
    {
        return __sum(v);
    }
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct lp_norm<T,2,false NBL_PARTIAL_REQ_BOT(conceptsconcepts::FloatingPointLikeVectorial<T>) >
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

// TODO: even/odd cases
}

template<typename T, uint32_t LP NBL_FUNC_REQUIRES(LP>0)
scalar_type_t<T> lpNormPreroot(NBL_CONST_REF_ARG(T) v)
{
    return impl::lp_norm<T,LP>::__sum(v);
}

template<typename T, uint32_t LP>
scalar_type_t<T> lpNorm(NBL_CONST_REF_ARG(T) v)
{
    return impl::lp_norm<T,LP>::__call(v);
}

template <typename T NBL_FUNC_REQUIRES(concepts::Vectorial<T> && vector_traits<T>::Dimension == 3)
T reflect(T I, T N, typename vector_traits<T>::scalar_type NdotI)
{
    return N * 2.0f * NdotI - I;
}

namespace impl
{
template<typename T>
struct orientedEtas;

template<>
struct orientedEtas<float>
{
    static bool __call(NBL_REF_ARG(float) orientedEta, NBL_REF_ARG(float) rcpOrientedEta, float NdotI, float eta)
    {
        const bool backside = NdotI < 0.0;
        const float rcpEta = 1.0 / eta;
        orientedEta = backside ? rcpEta : eta;
        rcpOrientedEta = backside ? eta : rcpEta;
        return backside;
    }
};

template<>
struct orientedEtas<float32_t3>
{
    static bool __call(NBL_REF_ARG(float32_t3) orientedEta, NBL_REF_ARG(float32_t3) rcpOrientedEta, float NdotI, float32_t3 eta)
    {
        const bool backside = NdotI < 0.0;
        const float32_t3 rcpEta = (float32_t3)1.0 / eta;
        orientedEta = backside ? rcpEta:eta;
        rcpOrientedEta = backside ? eta:rcpEta;
        return backside;
    }
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
bool getOrientedEtas(NBL_REF_ARG(T) orientedEta, NBL_REF_ARG(T) rcpOrientedEta, scalar_type_t<T> NdotI, T eta)
{
    return impl::orientedEtas<T>::__call(orientedEta, rcpOrientedEta, NdotI, eta);
}


namespace impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct refract;

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct refract<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) >
{
    using this_t = refract<T>;
    using vector_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    static this_t create(vector_type I, vector_type N, bool backside, scalar_type NdotI, scalar_type NdotI2, scalar_type rcpOrientedEta, scalar_type rcpOrientedEta2)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.backside = backside;
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI2;
        retval.rcpOrientedEta = rcpOrientedEta;
        retval.rcpOrientedEta2 = rcpOrientedEta2;
        return retval;
    }

    static this_t create(vector_type I, vector_type N, scalar_type NdotI, scalar_type eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        T orientedEta;
        retval.backside = getOrientedEtas<T>(orientedEta, retval.rcpOrientedEta, NdotI, eta);
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI * NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    static this_t create(vector_type I, vector_type N, scalar_type eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = dot<vector_type>(N, I);
        scalar_type orientedEta;
        retval.backside = getOrientedEtas<scalar_type>(orientedEta, retval.rcpOrientedEta, retval.NdotI, eta);
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    scalar_type computeNdotT()
    {
        scalar_type NdotT2 = rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        return backside ? absNdotT : -(absNdotT);
    }

    vector_type doRefract()
    {
        return N * (NdotI * rcpOrientedEta + computeNdotT()) - rcpOrientedEta * I;
    }

    static vector_type doReflectRefract(bool _refract, vector_type _I, vector_type _N, scalar_type _NdotI, scalar_type _NdotTorR, scalar_type _rcpOrientedEta)
    {    
        return _N * (_NdotI * (_refract ? _rcpOrientedEta : 1.0f) + _NdotTorR) - _I * (_refract ? _rcpOrientedEta : 1.0f);
    }

    vector_type doReflectRefract(bool r)
    {
        const scalar_type NdotTorR = r ? computeNdotT() : NdotI;
        return doReflectRefract(r, I, N, NdotI, NdotTorR, rcpOrientedEta);
    }

    vector_type I;
    vector_type N;
    bool backside;
    scalar_type NdotI;
    scalar_type NdotI2;
    scalar_type rcpOrientedEta;
    scalar_type rcpOrientedEta2;
};
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T refract(T I, T N, bool backside, 
    typename vector_traits<T>::scalar_type NdotI, 
    typename vector_traits<T>::scalar_type NdotI2,
    typename vector_traits<T>::scalar_type rcpOrientedEta,
    typename vector_traits<T>::scalar_type rcpOrientedEta2)
{
    impl::refract<T> r = impl::refract<T>::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doRefract();
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T refract(T I, T N, typename vector_traits<T>::scalar_type NdotI, typename vector_traits<T>::scalar_type eta)
{
    impl::refract<T> r = impl::refract<T>::create(I, N, NdotI, eta);
    return r.doRefract();
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T refract(T I, T N, typename vector_traits<T>::scalar_type eta)
{
    impl::refract<T> r = impl::refract<T>::create(I, N, eta);
    return r.doRefract();
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
typename vector_traits<T>::scalar_type reflectRefract_computeNdotT(bool backside, typename vector_traits<T>::scalar_type NdotI2, typename vector_traits<T>::scalar_type rcpOrientedEta2)
{
    impl::refract<T> r;
    r.NdotI2 = NdotI2;
    r.rcpOrientedEta2 = rcpOrientedEta2;
    r.backside = backside;
    return r.computeNdotT();
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T reflectRefract_impl(bool _refract, T _I, T _N,
    typename vector_traits<T>::scalar_type _NdotI,
    typename vector_traits<T>::scalar_type _NdotTorR,
    typename vector_traits<T>::scalar_type _rcpOrientedEta)
{
    return impl::refract<T>::doReflectRefract(_refract, _I, _N, _NdotI, _NdotTorR, _rcpOrientedEta);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T reflectRefract(bool _refract, T I, T N, bool backside, 
    typename vector_traits<T>::scalar_type NdotI,
    typename vector_traits<T>::scalar_type NdotI2,
    typename vector_traits<T>::scalar_type rcpOrientedEta,
    typename vector_traits<T>::scalar_type rcpOrientedEta2)
{
    impl::refract<T> r = impl::refract<T>::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doReflectRefract(_refract);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T reflectRefract(bool _refract, T I, T N, typename vector_traits<T>::scalar_type NdotI, typename vector_traits<T>::scalar_type eta)
{
    impl::refract<T> r = impl::refract<T>::create(I, N, NdotI, eta);
    return r.doReflectRefract(_refract);
}

// valid only for `theta` in [-PI,PI]
template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
void sincos(T theta, NBL_REF_ARG(T) s, NBL_REF_ARG(T) c)
{
    c = cos<T>(theta);
    s = sqrt<T>(NBL_FP64_LITERAL(1.0)-c*c);
    s = ieee754::flipSign(s, theta < NBL_FP64_LITERAL(0.0));
}

template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
matrix<T, 2, 3> frisvad(vector<T, 3> n)
{
	const T a = NBL_FP64_LITERAL(1.0) / (NBL_FP64_LITERAL(1.0) + n.z);
	const T b = -n.x * n.y * a;
	return (n.z < -NBL_FP64_LITERAL(0.9999999)) ? matrix<T, 2, 3>(vector<T, 3>(0.0,-1.0,0.0), vector<T, 3>(-1.0,0.0,0.0)) :
        matrix<T, 2, 3>(vector<T, 3>(NBL_FP64_LITERAL(1.0)-n.x*n.x*a, b, -n.x), vector<T, 3>(b, NBL_FP64_LITERAL(1.0)-n.y*n.y*a, -n.y));
}

bool partitionRandVariable(float leftProb, NBL_REF_ARG(float) xi, NBL_REF_ARG(float) rcpChoiceProb)
{
#ifdef __HLSL_VERSION
    NBL_CONSTEXPR float NEXT_ULP_AFTER_UNITY = asfloat(0x3f800001u);
#else
    NBL_CONSTEXPR float32_t NEXT_ULP_AFTER_UNITY = bit_cast<float32_t>(0x3f800001u);
#endif
    const bool pickRight = xi >= leftProb * NEXT_ULP_AFTER_UNITY;

    // This is all 100% correct taking into account the above NEXT_ULP_AFTER_UNITY
    xi -= pickRight ? leftProb : 0.0;

    rcpChoiceProb = NBL_FP64_LITERAL(1.0) / (pickRight ? (NBL_FP64_LITERAL(1.0) - leftProb) : leftProb);
    xi *= rcpChoiceProb;

    return pickRight;
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
        const T condAbs = bit_cast<T>(bit_cast<UintOfTSize>(x) & (cond ? 0x7fFFffFFu : 0xffFFffFFu));

        return max(condAbs, limit);
    }
};

template <typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct conditionalAbsOrMax_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) >
{
    static T __call(bool cond, NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) limit)
    {
        using UintOfTSize = unsigned_integer_of_size_t<sizeof(vector_traits<T>::scalar_type)>;
        const int dimensionOfT = vector_traits<T>::Dimension;
        using Uint32VectorWithDimensionOfT = vector<uint32_t, dimensionOfT>;
        using scalar_type = typename vector_traits<T>::scalar_type;

        Uint32VectorWithDimensionOfT xAsUintVec;
        {
            array_get<T, scalar_type> getter;
            array_set<Uint32VectorWithDimensionOfT, UintOfTSize> setter;

            for (int i = 0; i < dimensionOfT; ++i)
                setter(xAsUintVec, i, bit_cast<UintOfTSize>(getter(x, i)));
        }

        const Uint32VectorWithDimensionOfT mask = cond ? _static_cast<Uint32VectorWithDimensionOfT>(0x7fFFffFFu) : _static_cast<Uint32VectorWithDimensionOfT>(0xffFFffFFu);
        const Uint32VectorWithDimensionOfT condAbsAsUint = xAsUintVec & mask;

        T condAbs;
        {
            array_get<Uint32VectorWithDimensionOfT, UintOfTSize> getter;
            array_set<T, scalar_type> setter;

            for (int i = 0; i < dimensionOfT; ++i)
                setter(condAbs, i, bit_cast<scalar_type>(getter(condAbsAsUint, i)));
        }

        return max(condAbs, limit);
    }
};

struct trigonometry
{
    using this_t = trigonometry;

    static this_t create()
    {
        this_t retval;
        retval.tmp0 = 0;
        retval.tmp1 = 0;
        retval.tmp2 = 0;
        retval.tmp3 = 0;
        retval.tmp4 = 0;
        retval.tmp5 = 0;
        return retval;
    }

    static this_t create(float cosA, float cosB, float cosC, float sinA, float sinB, float sinC)
    {
        this_t retval;
        retval.tmp0 = cosA;
        retval.tmp1 = cosB;
        retval.tmp2 = cosC;
        retval.tmp3 = sinA;
        retval.tmp4 = sinB;
        retval.tmp5 = sinC;
        return retval;
    }

    float getArccosSumofABC_minus_PI()
    {
        const bool AltminusB = tmp0 < (-tmp1);
        const float cosSumAB = tmp0 * tmp1 - tmp3 * tmp4;
        const bool ABltminusC = cosSumAB < (-tmp2);
        const bool ABltC = cosSumAB < tmp2;
        // apply triple angle formula
        const float absArccosSumABC = acos<float>(clamp<float>(cosSumAB * tmp2 - (tmp0 * tmp4 + tmp3 * tmp1) * tmp5, -1.f, 1.f));
        return ((AltminusB ? ABltC : ABltminusC) ? (-absArccosSumABC) : absArccosSumABC) + (AltminusB | ABltminusC ? numbers::pi<float> : (-numbers::pi<float>));
    }

    static void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, NBL_REF_ARG(float) out0, NBL_REF_ARG(float) out1)
    {
        const float bias = biasA + biasB;
        const float a = cosA;
        const float b = cosB;
        const bool reverse = abs<float>(min<float>(a, b)) > max<float>(a, b);
        const float c = a * b - sqrt<float>((1.0f - a * a) * (1.0f - b * b));

        if (reverse)
        {
            out0 = -c;
            out1 = bias + numbers::pi<float>;
        }
        else
        {
            out0 = c;
            out1 = bias;
        }
    }

    float tmp0;
    float tmp1;
    float tmp2;
    float tmp3;
    float tmp4;
    float tmp5;
};
}

// @ return abs(x) if cond==true, max(x,0.0) otherwise
template <typename T>
T conditionalAbsOrMax(bool cond, T x, T limit)
{
    return impl::conditionalAbsOrMax_helper<T>::__call(cond, x, limit);
}

float getArccosSumofABC_minus_PI(float cosA, float cosB, float cosC, float sinA, float sinB, float sinC)
{
    impl::trigonometry trig = impl::trigonometry::create(cosA, cosB, cosC, sinA, sinB, sinC);
    return trig.getArccosSumofABC_minus_PI();
}

void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, NBL_REF_ARG(float) out0, NBL_REF_ARG(float) out1)
{
    impl::trigonometry trig = impl::trigonometry::create();
    impl::trigonometry::combineCosForSumOfAcos(cosA, cosB, biasA, biasB, trig.tmp0, trig.tmp1);
    out0 = trig.tmp0;
    out1 = trig.tmp1;
}

// returns acos(a) + acos(b)
float getSumofArccosAB(float cosA, float cosB)
{
    impl::trigonometry trig = impl::trigonometry::create();
    impl::trigonometry::combineCosForSumOfAcos(cosA, cosB, 0.0f, 0.0f, trig.tmp0, trig.tmp1);
    return acos<float>(trig.tmp0) + trig.tmp1;
}

// returns acos(a) + acos(b) + acos(c) + acos(d)
float getSumofArccosABCD(float cosA, float cosB, float cosC, float cosD)
{
    impl::trigonometry trig = impl::trigonometry::create();
    impl::trigonometry::combineCosForSumOfAcos(cosA, cosB, 0.0f, 0.0f, trig.tmp0, trig.tmp1);
    impl::trigonometry::combineCosForSumOfAcos(cosC, cosD, 0.0f, 0.0f, trig.tmp2, trig.tmp3);
    impl::trigonometry::combineCosForSumOfAcos(trig.tmp0, trig.tmp2, trig.tmp1, trig.tmp3, trig.tmp4, trig.tmp5);
    return acos<float>(trig.tmp4) + trig.tmp5;
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
