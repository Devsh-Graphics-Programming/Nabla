// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include "nbl/builtin/hlsl/ieee754.hlsl"
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

namespace fresnel
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct OrientedEtas
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static OrientedEtas<T> create(scalar_type NdotI, T eta)
    {
        OrientedEtas<T> retval;
        retval.backside = NdotI < scalar_type(0.0);
        const T rcpEta = hlsl::promote<T>(1.0) / eta;
        retval.value = retval.backside ? rcpEta : eta;
        retval.rcp = retval.backside ? eta : rcpEta;
        return retval;
    }

    T value;
    T rcp;
    bool backside;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct OrientedEtaRcps
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static OrientedEtaRcps<T> create(scalar_type NdotI, T eta)
    {
        OrientedEtaRcps<T> retval;
        retval.backside = NdotI < scalar_type(0.0);
        const T rcpEta = hlsl::promote<T>(1.0) / eta;
        retval.value = retval.backside ? eta : rcpEta;
        retval.value2 = retval.value * retval.value;
        return retval;
    }

    T diffuseFresnelCorrectionFactor()
    {
        // assert(n*n==n2);
        const T n = hlsl::promote<T>(1.0) / value;
        const T n2 = hlsl::promote<T>(1.0) / value2;

        vector<bool,vector_traits<T>::Dimension> TIR = n < hlsl::promote<T>(1.0);
        T invdenum = nbl::hlsl::mix<T>(hlsl::promote<T>(1.0), hlsl::promote<T>(1.0) / (n2 * n2 * (hlsl::promote<T>(554.33) - hlsl::promote<T>(380.7) * n)), TIR);
        T num = n * nbl::hlsl::mix<T>(hlsl::promote<T>(0.1921156102251088), n * hlsl::promote<T>(298.25) - hlsl::promote<T>(261.38) * n2 + hlsl::promote<T>(138.43), TIR);
        num += nbl::hlsl::mix<T>(hlsl::promote<T>(0.8078843897748912), hlsl::promote<T>(-1.67), TIR);
        return num * invdenum;
    }

    T value;
    T value2;
    bool backside;
};

}

template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct Reflect
{
    using this_t = Reflect<T>;
    using vector_type = vector<T, 3>;
    using scalar_type = T;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type NdotI)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = NdotI;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = computeNdotI(I, N);
        return retval;
    }

    void recomputeNdotI()
    {
        NdotI = hlsl::dot<vector_type>(N, I);
    }

    vector_type operator()()
    {
        return N * 2.0f * NdotI - I;
    }

    vector_type I;
    vector_type N;
    scalar_type NdotI;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct Refract
{
    using this_t = Refract<T>;
    using vector_type = vector<T, 3>;
    using scalar_type = T;

    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<scalar_type>) rcpEtas, NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = hlsl::dot<vector_type>(N, I);
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta = rcpEtas;
        retval.recomputeNdotT(rcpEtas.backside, retval.NdotI2, rcpEtas.value2);
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<scalar_type>) rcpEtas, NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, const scalar_type NdotI)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = NdotI;
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta = rcpEtas;
        retval.recomputeNdotT(rcpEtas.backside, retval.NdotI2, rcpEtas.value2);
        return retval;
    }

    void recomputeNdotI()
    {
        NdotI = hlsl::dot<vector_type>(N, I);
        NdotI2 = NdotI * NdotI;
    }

    void recomputeNdotT(bool backside, scalar_type _NdotI2, scalar_type rcpOrientedEta2)
    {
        scalar_type NdotT2 = rcpOrientedEta2 * _NdotI2 + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        NdotT = ieee754::flipSign(absNdotT, backside);
    }

    vector_type operator()()
    {
        recomputeNdotT(rcpOrientedEta.backside, NdotI2, rcpOrientedEta.value2);
        return N * (NdotI * rcpOrientedEta.value + NdotT) - rcpOrientedEta.value * I;
    }

    vector_type I;
    vector_type N;
    scalar_type NdotT;
    scalar_type NdotI;
    scalar_type NdotI2;
    fresnel::OrientedEtaRcps<scalar_type> rcpOrientedEta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct ReflectRefract
{
    using this_t = ReflectRefract<T>;
    using vector_type = vector<T, 3>;
    using scalar_type = T;

    static this_t create(bool refract, NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type NdotI, scalar_type NdotTorR, scalar_type rcpOrientedEta)
    {
        this_t retval;
        retval.is_refract = refract;
        retval.I = I;
        retval.N = N;
        retval.NdotI = NdotI;
        retval.NdotTorR = NdotTorR;
        retval.rcpOrientedEta = rcpOrientedEta;
        return retval;
    }

    static this_t create(bool r, NBL_CONST_REF_ARG(Refract<scalar_type>) refract)
    {
        this_t retval;
        retval.is_refract = r;
        retval.I = refract.I;
        retval.N = refract.N;
        retval.NdotI = refract.NdotI;
        retval.NdotTorR = hlsl::mix(refract.NdotI, refract.NdotT, r);
        retval.rcpOrientedEta = refract.rcpOrientedEta.value;
        return retval;
    }

    vector_type operator()()
    {
        return N * (NdotI * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, is_refract)) + NdotTorR) - I * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, is_refract));
    }

    bool is_refract;
    Refract<scalar_type> refract;
    vector_type I;
    vector_type N;
    scalar_type NdotI;
    scalar_type NdotTorR;
    scalar_type rcpOrientedEta;
};


namespace fresnel
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct Schlick
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Schlick<T> create(NBL_CONST_REF_ARG(T) F0, scalar_type VdotH)
    {
        Schlick<T> retval;
        retval.F0 = F0;
        retval.VdotH = VdotH;
        return retval;
    }

    T operator()()
    {
        T x = 1.0 - VdotH;
        return F0 + (1.0 - F0) * x*x*x*x*x;
    }

    T F0;
    scalar_type VdotH;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct Conductor
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Conductor<T> create(NBL_CONST_REF_ARG(T) eta, NBL_CONST_REF_ARG(T) etak, scalar_type cosTheta)
    {
        Conductor<T> retval;
        retval.eta = eta;
        retval.etak = etak;
        retval.cosTheta = cosTheta;
        return retval;
    }

    T operator()()
    {
        const scalar_type cosTheta2 = cosTheta * cosTheta;
        //const float sinTheta2 = 1.0 - cosTheta2;

        const T etaLen2 = eta * eta + etak * etak;
        const T etaCosTwice = eta * cosTheta * 2.0f;

        const T rs_common = etaLen2 + (T)(cosTheta2);
        const T rs2 = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);

        const T rp_common = etaLen2 * cosTheta2 + (T)(1.0);
        const T rp2 = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);

        return (rs2 + rp2) * 0.5f;
    }

    T eta;
    T etak;
    scalar_type cosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct Dielectric
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Dielectric<T> create(NBL_CONST_REF_ARG(T) eta, scalar_type cosTheta)
    {
        Dielectric<T> retval;
        OrientedEtas<T> orientedEta = OrientedEtas<T>::create(cosTheta, eta);
        retval.eta2 = orientedEta.value * orientedEta.value;
        retval.cosTheta = cosTheta;
        return retval;
    }

    static T __call(NBL_CONST_REF_ARG(T) orientedEta2, scalar_type absCosTheta)
    {
        const scalar_type sinTheta2 = 1.0 - absCosTheta * absCosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        const T t0 = hlsl::sqrt<T>(hlsl::max<T>(orientedEta2 - sinTheta2, hlsl::promote<T>(0.0)));
        const T rs = (hlsl::promote<T>(absCosTheta) - t0) / (hlsl::promote<T>(absCosTheta) + t0);

        const T t2 = orientedEta2 * absCosTheta;
        const T rp = (t0 - t2) / (t0 + t2);

        return (rs * rs + rp * rp) * 0.5f;
    }

    T operator()()
    {
        return __call(eta2, cosTheta);
    }

    T eta2;
    scalar_type cosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
struct DielectricFrontFaceOnly
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static DielectricFrontFaceOnly<T> create(NBL_CONST_REF_ARG(T) orientedEta2, scalar_type absCosTheta)
    {
        Dielectric<T> retval;
        retval.orientedEta2 = orientedEta2;
        retval.absCosTheta = hlsl::abs<T>(absCosTheta);
        return retval;
    }

    T operator()()
    {
        return Dielectric<T>::__call(orientedEta2, absCosTheta);
    }

    T orientedEta2;
    scalar_type absCosTheta;
};


// gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
template<typename T>
struct ThinDielectricInfiniteScatter
{
    T operator()(T singleInterfaceReflectance)
    {
        const T doubleInterfaceReflectance = singleInterfaceReflectance * singleInterfaceReflectance;
        return hlsl::mix<T>(hlsl::promote<T>(1.0), (singleInterfaceReflectance - doubleInterfaceReflectance) / (hlsl::promote<T>(1.0) - doubleInterfaceReflectance) * 2.0f, doubleInterfaceReflectance > hlsl::promote<T>(0.9999));
    }
};

}

}
}
}

#endif
