// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

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

    static T diffuseFresnelCorrectionFactor(T n, T n2)
    {
        // assert(n*n==n2);
        vector<bool,vector_traits<T>::Dimension> TIR = n < (T)1.0;
        T invdenum = nbl::hlsl::mix<T>(hlsl::promote<T>(1.0), hlsl::promote<T>(1.0) / (n2 * n2 * (hlsl::promote<T>(554.33) - 380.7 * n)), TIR);
        T num = n * nbl::hlsl::mix<T>(hlsl::promote<T>(0.1921156102251088), n * 298.25 - 261.38 * n2 + 138.43, TIR);
        num += nbl::hlsl::mix<T>(hlsl::promote<T>(0.8078843897748912), hlsl::promote<T>(-1.67), TIR);
        return num * invdenum;
    }

    T value;
    T rcp;
    bool backside;
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


template <typename T NBL_FUNC_REQUIRES(concepts::Vectorial<T>)
T reflect(T I, T N, typename vector_traits<T>::scalar_type NdotI)
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
        const T NdotTorR = r ? computeNdotT(backside, NdotI2, rcpOrientedEta2) : NdotI;
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

template<typename T NBL_PRIMARY_REQUIRES(concepts::Vectorial<T> && vector_traits<T>::Dimension == 3)
struct ReflectRefract
{
    using this_t = ReflectRefract<T>;
    using vector_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    static this_t create(bool refract, NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type NdotI, scalar_type NdotTorR, scalar_type rcpOrientedEta)
    {
        this_t retval;
        retval.refract = refract;
        retval.I = I;
        retval.N = N;
        retval.NdotI = NdotI;
        retval.NdotTorR = NdotTorR;
        retval.rcpOrientedEta = rcpOrientedEta;
        return retval;
    }

    static this_t create(bool r, NBL_CONST_REF_ARG(Refract<vector_type>) refract)
    {
        this_t retval;
        retval.refract = r;
        retval.I = refract.I;
        retval.N = refract.N;
        retval.NdotI = refract.NdotI;
        retval.NdotTorR = r ? Refract<vector_type>::computeNdotT(refract.backside, refract.NdotI2, refract.rcpOrientedEta2) : refract.NdotI;
        retval.rcpOrientedEta = refract.rcpOrientedEta;
        return retval;
    }

    vector_type operator()()
    {
        return N * (NdotI * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, refract)) + NdotTorR) - I * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, refract));
    }

    bool refract;
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

#endif
