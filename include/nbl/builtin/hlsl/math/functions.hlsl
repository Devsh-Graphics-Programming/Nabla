// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{
template<typename T, uint32_t LP, bool Odd=(LP&0x1)>
struct lp_norm;

// infinity case
template<typename T>
struct lp_norm<T,0,false>
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __call(const T v)
    {
        scalar_type retval = nbl::hlsl::abs<T>(v[0]);
        for (int i = 1; i < extent<T>::value; i++)
            retval = nbl::hlsl::max<T>(nbl::hlsl::abs<T>(v[i]),retval);
        return retval;
    }
};

template<typename T>
struct lp_norm<T,1,true>
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __sum(const T v)
    {
        scalar_type retval = nbl::hlsl::abs<T>(v[0]);
        for (int i = 1; i < extent<T>::value; i++)
            retval += nbl::hlsl::abs<T>(v[i]);
        return retval;
    }

    static scalar_type __call(const T v)
    {
        return __sum(v);
    }
};

template<typename T>
struct lp_norm<T,2,false>
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static scalar_type __sum(const T v)
    {
        return nbl::hlsl::dot<T>(v, v);
    }

    static scalar_type __call(const T v)
    {
        return nbl::hlsl::sqrt<scalar_type>(__sum(v));
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
template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
void sincos(T theta, NBL_REF_ARG(T) s, NBL_REF_ARG(T) c)
{
    c = cos<T>(theta);
    s = sqrt<T>(1.0-c*c);
    s = (theta < 0.0) ? -s : s; // TODO: test with XOR
}

template <typename T NBL_FUNC_REQUIRES(vector_traits<T>::Dimension == 3)
void frisvad(NBL_CONST_REF_ARG(T) normal, NBL_REF_ARG(T) tangent, NBL_REF_ARG(T) bitangent)
{
	const typename vector_traits<T>::scalar_type a = 1.0 / (1.0 + normal.z);
	const typename vector_traits<T>::scalar_type b = -normal.x * normal.y * a;
    if (normal.z < -0.9999999)
    {
        tangent = T(0.0,-1.0,0.0);
        bitangent = T(-1.0,0.0,0.0);
    }
    else
    {
        tangent = T(1.0-normal.x*normal.x*a, b, -normal.x);
        bitangent = T(b, 1.0-normal.y*normal.y*a, -normal.y);
    }
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

    rcpChoiceProb = 1.0 / (pickRight ? (1.0 - leftProb) : leftProb);
    xi *= rcpChoiceProb;

    return pickRight;
}


// TODO: impl signed integer versions
// @ return abs(x) if cond==true, max(x,0.0) otherwise
template <typename T NBL_FUNC_REQUIRES(is_floating_point_v<T> || concepts::FloatingPointVector<T> || concepts::FloatingPointVectorial<T>)
T conditionalAbsOrMax(bool cond, T x, T limit);

template <>
float conditionalAbsOrMax<float>(bool cond, float x, float limit)
{
    const float condAbs = nbl::hlsl::bit_cast<float32_t, uint32_t>(nbl::hlsl::bit_cast<uint32_t, float32_t>(x) & uint(cond ? 0x7fFFffFFu : 0xffFFffFFu));
    return nbl::hlsl::max<float>(condAbs,limit);
}

template <uint16_t N>
vector<float, N> conditionalAbsOrMax<vector<float, N> >(bool cond, NBL_CONST_REF_ARG(vector<float, N>) x, NBL_CONST_REF_ARG(vector<float, N>) limit)
{
    const vector<float, N> condAbs = nbl::hlsl::bit_cast<vector<float, N>, vector<uint, N> >(nbl::hlsl::bit_cast<vector<uint, N>, vector<float, N> >(x) & nbl::hlsl::mix((vector<uint, N>)0x7fFFffFFu, (vector<uint, N>)0xffFFffFFu, promote<vector<bool, N>, bool>(cond)));
    return nbl::hlsl::max<vector<float, N> >(condAbs,limit);
}


namespace impl
{
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

template<typename T, uint16_t M, uint16_t N, uint16_t P NBL_FUNC_REQUIRES(is_scalar_v<T>)
matrix<T,M,P> applyChainRule(NBL_CONST_REF_ARG(matrix<T,N,M>) dFdG, NBL_CONST_REF_ARG(matrix<T,M,P>) dGdR)
{
    return mul(dFdG,dGdR);
}

}
}
}

#endif
