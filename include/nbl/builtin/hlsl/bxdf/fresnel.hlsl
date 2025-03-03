// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace bxdf
{

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


template<typename T NBL_FUNC_REQUIRES(concepts::Vectorial<T>)
T reflect(NBL_CONST_REF_ARG(T) I, NBL_CONST_REF_ARG(T) N, typename vector_traits<T>::scalar_type NdotI)
{
    return N * 2.0f * NdotI - I;
}

template<typename T NBL_PRIMARY_REQUIRES(vector_traits<T>::Dimension == 3)
struct refract
{
    using this_t = refract<T>;
    using vector_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, bool backside, scalar_type NdotI, scalar_type NdotI2, scalar_type rcpOrientedEta, scalar_type rcpOrientedEta2)
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

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type NdotI, scalar_type eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        scalar_type orientedEta;
        retval.backside = getOrientedEtas<scalar_type>(orientedEta, retval.rcpOrientedEta, NdotI, eta);
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI * NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type eta)
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

    static scalar_type computeNdotT(bool backside, scalar_type NdotI2, scalar_type rcpOrientedEta2)
    {
        scalar_type NdotT2 = rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        return backside ? absNdotT : -(absNdotT);
    }

    vector_type doRefract()
    {
        return N * (NdotI * rcpOrientedEta + computeNdotT(backside, NdotI2, rcpOrientedEta2)) - rcpOrientedEta * I;
    }

    static vector_type doReflectRefract(bool _refract, NBL_CONST_REF_ARG(vector_type) _I, NBL_CONST_REF_ARG(vector_type) _N, scalar_type _NdotI, scalar_type _NdotTorR, scalar_type _rcpOrientedEta)
    {
        return _N * (_NdotI * (_refract ? _rcpOrientedEta : 1.0f) + _NdotTorR) - _I * (_refract ? _rcpOrientedEta : 1.0f);
    }

    vector_type doReflectRefract(bool r)
    {
        const scalar_type NdotTorR = r ? computeNdotT(backside, NdotI2, rcpOrientedEta2) : NdotI;
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


template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct fresnel
{
    using scalar_t = typename scalar_type<T>::type;

    static T schlick(NBL_CONST_REF_ARG(T) F0, scalar_t VdotH)
    {
        T x = 1.0 - VdotH;
        return F0 + (1.0 - F0) * x*x*x*x*x;
    }

    static T conductor(NBL_CONST_REF_ARG(T) eta, NBL_CONST_REF_ARG(T) etak, scalar_t cosTheta)
    {
        const scalar_t cosTheta2 = cosTheta * cosTheta;
        //const float sinTheta2 = 1.0 - cosTheta2;

        const T etaLen2 = eta * eta + etak * etak;
        const T etaCosTwice = eta * cosTheta * 2.0f;

        const T rs_common = etaLen2 + (T)(cosTheta2);
        const T rs2 = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);

        const T rp_common = etaLen2 * cosTheta2 + (T)(1.0);
        const T rp2 = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);

        return (rs2 + rp2) * 0.5f;
    }

    static T dielectric_common(NBL_CONST_REF_ARG(T) orientedEta2, scalar_t absCosTheta)
    {
        const scalar_t sinTheta2 = 1.0 - absCosTheta * absCosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        const T t0 = nbl::hlsl::sqrt<T>(nbl::hlsl::max<T>((T)(orientedEta2) - sinTheta2, (T)(0.0)));
        const T rs = ((T)(absCosTheta) - t0) / ((T)(absCosTheta) + t0);

        const T t2 = orientedEta2 * absCosTheta;
        const T rp = (t0 - t2) / (t0 + t2);

        return (rs * rs + rp * rp) * 0.5f;
    }

    static T dielectricFrontFaceOnly(NBL_CONST_REF_ARG(T) orientedEta2, scalar_t absCosTheta)
    {
        return dielectric_common(orientedEta2, absCosTheta);
    }

    static T dielectric(NBL_CONST_REF_ARG(T) eta, scalar_t cosTheta)
    {
        T orientedEta, rcpOrientedEta;
        bxdf::getOrientedEtas<T>(orientedEta, rcpOrientedEta, cosTheta, eta);
        return dielectric_common(orientedEta * orientedEta, cosTheta);
    }
};

namespace impl
{
// gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
template<typename T>
struct ThinDielectricInfiniteScatter
{
    using scalar_t = typename scalar_type<T>::type;

    static T __call(T singleInterfaceReflectance)
    {
        const T doubleInterfaceReflectance = singleInterfaceReflectance * singleInterfaceReflectance;
        return nbl::hlsl::mix<T>((singleInterfaceReflectance - doubleInterfaceReflectance) / ((T)(1.0) - doubleInterfaceReflectance) * 2.0f, (T)(1.0), doubleInterfaceReflectance > (T)(0.9999));
    }
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T thindielectricInfiniteScatter(T singleInterfaceReflectance)
{
    return impl::ThinDielectricInfiniteScatter<T>::__call(singleInterfaceReflectance);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T diffuseFresnelCorrectionFactor(T n, T n2)
{
    // assert(n*n==n2);
    vector<bool,vector_traits<T>::Dimension> TIR = n < (T)1.0;
    T invdenum = nbl::hlsl::mix<T>((T)1.0, (T)1.0 / (n2 * n2 * ((T)554.33 - 380.7 * n)), TIR);
    T num = n * nbl::hlsl::mix<T>((T)(0.1921156102251088), n * 298.25 - 261.38 * n2 + 138.43, TIR);
    num += nbl::hlsl::mix<T>((T)(0.8078843897748912), (T)(-1.67), TIR);
    return num * invdenum;
}

}
}
}

#endif
