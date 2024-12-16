// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
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
    static scalar_type_t<T> __call(const T v)
    {
        scalar_type_t<T> retval = abs<T>(v[0]);
        for (int i = 1; i < rank<T>::value; i++)
            retval = max<T>(abs<T>(v[i]),retval);
        return retval;
    }
};

// TOOD: is this doing what it should be?
template<typename T>
struct lp_norm<T,1,false>
{
    static scalar_type_t<T> __sum(const T v)
    {
        scalar_type_t<T> retval = abs<T>(v[0]);
        for (int i = 1; i < rank<T>::value; i++)
            retval += abs<T>(v[i]);
        return retval;
    }

    static scalar_type_t<T> __call(const T v)
    {
        return __sum(v);
    }
};

template<typename T>
struct lp_norm<T,2,false>
{
    static scalar_type_t<T> __sum(const T v)
    {
        return dot<T>(v, v);   // TODO: wait for overloaded dot?
    }

    static scalar_type_t<T> __call(const T v)
    {
        return sqrt<T>(__sum(v));
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


template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T, 3> reflect(vector<T, 3> I, vector<T, 3> N, T NdotI)
{
    return N * 2.0 * NdotI - I;
}

template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T, 3> reflect(vector<T, 3> I, vector<T, 3> N)
{
    T NdotI = dot<T>(N, I);
    return reflect<T>(I, N, NdotI);
}


namespace impl
{
template<typename T>
struct orientedEtas;

template<>
struct orientedEtas<float>
{
    static bool __call(out float orientedEta, out float rcpOrientedEta, float NdotI, float eta)
    {
        const bool backside = NdotI < 0.0;
        const float rcpEta = 1.0 / eta;
        orientedEta = backside ? rcpEta : eta;
        rcpOrientedEta = backside ? eta : rcpEta;
        return backside;
    }
};

template<>
struct orientedEtas<float3>
{
    static bool __call(out float3 orientedEta, out float3 rcpOrientedEta, float NdotI, float3 eta)
    {
        const bool backside = NdotI < 0.0;
        const float3 rcpEta = (float3)1.0 / eta;
        orientedEta = backside ? rcpEta:eta;
        rcpOrientedEta = backside ? eta:rcpEta;
        return backside;
    }
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
bool getOrientedEtas(out T orientedEta, out T rcpOrientedEta, scalar_type_t<T> NdotI, T eta)
{
    return impl::orientedEtas<T>::__call(orientedEta, rcpOrientedEta, NdotI, eta);
}


namespace impl
{
template<typename T>
struct refract
{
    using this_t = refract;
    using vector_type = vector<T,3>;

    static this_t create(vector_type I, vector_type N, bool backside, T NdotI, T NdotI2, T rcpOrientedEta, T rcpOrientedEta2)
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

    static this_t create(vector_type I, vector_type N, T NdotI, T eta)
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

    static this_t create(vector_type I, vector_type N, T eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = dot<T>(N, I);
        T orientedEta;
        retval.backside = getOrientedEtas<T>(orientedEta, retval.rcpOrientedEta, retval.NdotI, eta);        
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    T computeNdotT()
    {
        T NdotT2 = rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
        T absNdotT = sqrt<T>(NdotT2);
        return backside ? absNdotT : -(absNdotT);
    }

    vector_type doRefract()
    {
        return N * (NdotI * rcpOrientedEta + computeNdotT()) - rcpOrientedEta * I;
    }

    static vector_type doReflectRefract(bool _refract, vector_type _I, vector_type _N, T _NdotI, T _NdotTorR, T _rcpOrientedEta)
    {    
        return _N * (_NdotI * (_refract ? _rcpOrientedEta : 1.0) + _NdotTorR) - _I * (_refract ? _rcpOrientedEta : 1.0);
    }

    vector_type doReflectRefract(bool r)
    {
        const T NdotTorR = r ? computeNdotT() : NdotI;
        return doReflectRefract(r, I, N, NdotI, NdotTorR, rcpOrientedEta);
    }

    vector_type I;
    vector_type N;
    bool backside;
    T NdotI;
    T NdotI2;
    T rcpOrientedEta;
    T rcpOrientedEta2;
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> refract(vector<T,3> I, vector<T,3> N, bool backside, T NdotI, T NdotI2, T rcpOrientedEta, T rcpOrientedEta2)
{
    impl::refract r = impl::refract::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doRefract();
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> refract(vector<T,3> I, vector<T,3> N, T NdotI, T eta)
{
    impl::refract r = impl::refract::create(I, N, NdotI, eta);
    return r.doRefract();
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> refract(vector<T,3> I, vector<T,3> N, T eta)
{
    impl::refract r = impl::refract::create(I, N, eta);
    return r.doRefract();
}

// I don't like exposing these next two
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> reflectRefract_computeNdotT(bool backside, T NdotI2, T rcpOrientedEta2)
{
    impl::refract r;
    r.NdotI2 = NdotI2;
    r.rcpOrientedEta2 = rcpOrientedEta2;
    r.backside = backside;
    return r.computeNdotT();
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> reflectRefract_impl(bool _refract, vector<T,3> _I, vector<T,3> _N, T _NdotI, T _NdotTorR, T _rcpOrientedEta)
{
    return impl::refract::doReflectRefract(_refract, _I, _N, _NdotI, _NdotTorR, _rcpOrientedEta);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> reflectRefract(bool _refract, vector<T,3> I, vector<T,3> N, bool backside, T NdotI, T NdotI2, T rcpOrientedEta, T rcpOrientedEta2)
{
    impl::refract r = impl::refract::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doReflectRefract(_refract);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
vector<T,3> reflectRefract(bool _refract, vector<T,3> I, vector<T,3> N, T NdotI, T eta)
{
    impl::refract r = impl::refract::create(I, N, NdotI, eta);
    return r.doReflectRefract(_refract);
}


// valid only for `theta` in [-PI,PI]
template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
void sincos(T theta, out T s, out T c)
{
    c = cos<T>(theta);
    s = sqrt<T>(1.0-c*c);
    s = (theta < 0.0) ? -s : s; // TODO: test with XOR
}

template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
matrix<T, 3, 2> frisvad(vector<T, 3> n) // TODO: confirm dimensions of matrix
{
	const T a = 1.0 / (1.0 + n.z);
	const T b = -n.x * n.y * a;
	return (n.z < -0.9999999) ? matrix<T, 2, 3>(vector<T, 3>(0.0,-1.0,0.0), vector<T, 3>(-1.0,0.0,0.0)) : 
        matrix<T, 2, 3>(vector<T, 3>(1.0-n.x*n.x*a, b, -n.x), vector<T, 3>(b, 1.0-n.y*n.y*a, -n.y));
}

bool partitionRandVariable(in float leftProb, inout float xi, out float rcpChoiceProb)
{
#ifdef __HLSL_VERSION
    NBL_CONSTEXPR float NEXT_ULP_AFTER_UNITY = asfloat(0x3f800001u);
#else
    NBL_CONSTEXPR uint32_t val = 0x3f800001u;
    NBL_CONSTEXPR float32_t NEXT_ULP_AFTER_UNITY = reinterpret_cast<float32_t &>( val );
#endif
    const bool pickRight = xi >= leftProb * NEXT_ULP_AFTER_UNITY;

    // This is all 100% correct taking into account the above NEXT_ULP_AFTER_UNITY
    xi -= pickRight ? leftProb : 0.0;

    rcpChoiceProb = 1.0 / (pickRight ? (1.0 - leftProb) : leftProb);
    xi *= rcpChoiceProb;

    return pickRight;
}


// TODO: make it work in C++, ignoring problem for now
#ifdef __HLSL_VERSION
// @ return abs(x) if cond==true, max(x,0.0) otherwise
template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T conditionalAbsOrMax(bool cond, T x, T limit);

template <>
float conditionalAbsOrMax<float>(bool cond, float x, float limit)
{
    const float condAbs = asfloat(asuint(x) & uint(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}

template <>
float2 conditionalAbsOrMax<float2>(bool cond, float2 x, float2 limit)
{
    const float2 condAbs = asfloat(asuint(x) & select(cond, (uint2)0x7fFFffFFu, (uint2)0xffFFffFFu));
    return max(condAbs,limit);
}

template <>
float3 conditionalAbsOrMax<float3>(bool cond, float3 x, float3 limit)
{
    const float3 condAbs = asfloat(asuint(x) & select(cond, (uint3)0x7fFFffFFu, (uint3)0xffFFffFFu));
    return max(condAbs,limit);
}

template <>
float4 conditionalAbsOrMax<float4>(bool cond, float4 x, float4 limit)
{
    const float4 condAbs = asfloat(asuint(x) & select(cond, (uint4)0x7fFFffFFu, (uint4)0xffFFffFFu));
    return max(condAbs,limit);
}
#endif

namespace impl
{
struct bitFields    // need template?
{
    using this_t = bitFields;

    static this_t create(uint base, uint value, uint offset, uint count)
    {
        this_t retval;
        retval.base = base;
        retval.value = value;
        retval.offset = offset;
        retval.count = count;
        return retval;
    }

    uint __insert()
    {
        const uint shifted_masked_value = (value & ((0x1u << count) - 1u)) << offset;
        const uint lo = base & ((0x1u << offset) - 1u);
        const uint hi = base ^ lo;
        return (hi << count) | shifted_masked_value | lo;
    }

    uint __overwrite()
    {
        return spirv::bitFieldInsert<uint>(base, value, offset, count);
    }

    uint base;
    uint value;
    uint offset;
    uint count;
};
}

uint bitFieldOverwrite(uint base, uint value, uint offset, uint count)
{
    impl::bitFields b = impl::bitFields::create(base, value, offset, count);
    return b.__overwrite();
}

uint bitFieldInsert(uint base, uint value, uint offset, uint count)
{
    impl::bitFields b = impl::bitFields::create(base, value, offset, count);
    return b.__insert();
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

    static void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, out float out0, out float out1)
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

void combineCosForSumOfAcos(float cosA, float cosB, float biasA, float biasB, out float out0, out float out1)
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

namespace impl
{
template<typename T, uint16_t M, uint16_t N, uint16_t P>
struct applyChainRule4D
{
    static matrix<T, P, M> __call(matrix<T, N, M> dFdG, matrix<T, P, N> dGdR)
    {
#ifdef __HLSL_VERSION
        return mul(dFdG, dGdR);
#else
        return dFdG * dGdR; // glm
#endif
    }
};

template<typename T, uint16_t M, uint16_t N>
struct applyChainRule3D : applyChainRule4D<T,M,N,1>
{
    static vector<T, N> __call(matrix<T, N, M> dFdG, vector<T, N> dGdR)
    {
#ifdef __HLSL_VERSION
        return mul(dFdG, dGdR);
#else
        return dFdG * dGdR; // glm
#endif
    }
};

template<typename T, uint16_t M>
struct applyChainRule2D : applyChainRule4D<T,M,1,1>
{
    static vector<T, M> __call(vector<T, M> dFdG, T dGdR)
    {
#ifdef __HLSL_VERSION
        return mul(dFdG, dGdR);
#else
        return dFdG * dGdR; // glm
#endif
    }
};

template<typename T>
struct applyChainRule1D : applyChainRule4D<T,1,1,1>
{
    static T __call(T dFdG, T dGdR)
    {
        return dFdG * dGdR;
    }
};
}

// possible to derive M,N,P automatically?
template<typename T, uint16_t M, uint16_t N, uint16_t P NBL_FUNC_REQUIRES(is_scalar_v<T> && M>1 && N>1 && P>1)
matrix<T, P, M> applyChainRule(matrix<T, N, M> dFdG, matrix<T, P, N> dGdR)
{
    return impl::applyChainRule4D<T,M,N,P>::__call(dFdG, dGdR);
}

template<typename T, uint16_t M, uint16_t N NBL_FUNC_REQUIRES(is_scalar_v<T> && M>1 && N>1)
vector<T, N> applyChainRule(matrix<T, N, M> dFdG, vector<T, N> dGdR)
{
    return impl::applyChainRule3D<T,M,N>::__call(dFdG, dGdR);
}

template<typename T, uint16_t M NBL_FUNC_REQUIRES(is_scalar_v<T> && M>1)
vector<T, M> applyChainRule(vector<T, M> dFdG, T dGdR)
{
    return impl::applyChainRule2D<T,M>::__call(dFdG, dGdR);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T applyChainRule(T dFdG, T dGdR)
{
    return impl::applyChainRule1D<T>::__call(dFdG, dGdR);
}

}
}
}

#endif
