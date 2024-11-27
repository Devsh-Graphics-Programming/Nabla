// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

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
        scalar_type_t<T> retval = abs(v[0]);
        for (int i = 1; i < dimensions_v<T>; i++)
            retval = max(abs(v[i]),retval);
        return retval;
    }
};

// TOOD: is this doing what it should be?
template<typename T>
struct lp_norm<T,1,false>
{
    static scalar_type_t<T> __sum(const T v)
    {
        scalar_type_t<T> retval = abs(v[0]);
        for (int i = 1; i < dimensions_v<T>; i++)
            retval += abs(v[i]);
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
        return dot(v, v);   // TODO: wait for overloaded dot?
    }

    static scalar_type_t<T> __call(const T v)
    {
        return sqrt(__sum(v));
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
    T NdotI = dot(N, I);
    return reflect(I, N, NdotI);
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
struct refract
{
    using this_t = refract;

    static this_t create(float3 I, float3 N, bool backside, float NdotI, float NdotI2, float rcpOrientedEta, float rcpOrientedEta2)
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

    static this_t create(float3 I, float3 N, float NdotI, float eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        float orientedEta;
        retval.backside = getOrientedEtas<float>(orientedEta, retval.rcpOrientedEta, NdotI, eta);
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI * NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    static this_t create(float3 I, float3 N, float eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = dot(N, I);
        float orientedEta;
        retval.backside = getOrientedEtas<float>(orientedEta, retval.rcpOrientedEta, retval.NdotI, eta);        
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    float computeNdotT()
    {
        float NdotT2 = rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
        float absNdotT = sqrt(NdotT2);
        return backside ? absNdotT : -(absNdotT);
    }

    float3 doRefract()
    {
        return N * (NdotI * rcpOrientedEta + computeNdotT()) - rcpOrientedEta * I;
    }

    float3 doReflectRefract(bool r)
    {
        const float NdotTorR = r ? computeNdotT(): NdotI;
        return N * (NdotI * (r ? rcpOrientedEta : 1.0) + NdotTorR) - I * (r ? rcpOrientedEta : 1.0);
    }

    float3 I;
    float3 N;
    bool backside;
    float NdotI;
    float NdotI2;
    float rcpOrientedEta;
    float rcpOrientedEta2;
};
}

float3 refract(float3 I, float3 N, bool backside, float NdotI, float NdotI2, float rcpOrientedEta, float rcpOrientedEta2)
{
    impl::refract r = impl::refract::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doRefract();
}

float3 refract(float3 I, float3 N, float NdotI, float eta)
{
    impl::refract r = impl::refract::create(I, N, NdotI, eta);
    return r.doRefract();
}

float3 refract(float3 I, float3 N, float eta)
{
    impl::refract r = impl::refract::create(I, N, eta);
    return r.doRefract();
}

float3 reflectRefract(bool _refract, float3 I, float3 N, bool backside, float NdotI, float NdotI2, float rcpOrientedEta, float rcpOrientedEta2)
{
    impl::refract r = impl::refract::create(I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
    return r.doReflectRefract(_refract);
}

float3 reflectRefract(bool _refract, float3 I, float3 N, float NdotI, float eta)
{
    impl::refract r = impl::refract::create(I, N, NdotI, eta);
    return r.doReflectRefract(_refract);
}


// valid only for `theta` in [-PI,PI]
template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
void sincos(T theta, out T s, out T c)
{
    c = cos(theta);
    s = sqrt(1.0-c*c);
    s = (theta < 0.0) ? -s : s; // TODO: test with XOR
}

template <typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
matrix<T, 2, 3> frisvad(vector<T, 3> n) // TODO: confirm dimensions of matrix
{
	const float a = 1.0 / (1.0 + n.z);
	const float b = -n.x * n.y * a;
	return (n.z < -0.9999999) ? matrix<T, 2, 3>(vector<T, 3>(0.0,-1.0,0.0), vector<T, 3>(-1.0,0.0,0.0)) : 
        matrix<T, 2, 3>(vector<T, 3>(1.0-n.x*n.x*a, b, -n.x), vector<T, 3>(b, 1.0-n.y*n.y*a, -n.y));
}

}
}
}

#endif
