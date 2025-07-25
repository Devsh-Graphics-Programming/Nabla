// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include "nbl/builtin/hlsl/ieee754.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace bxdf
{

namespace fresnel
{

// derived from the Earl Hammon GDC talk but improved for all IOR not just 1.333 with actual numerical integration and curve fitting
template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct DiffuseCorrectionFactor
{
    T operator()()
    {
        vector<bool,vector_traits<T>::Dimension> TIR = orientedEta < hlsl::promote<T>(1.0);
        T invdenum = nbl::hlsl::mix<T>(hlsl::promote<T>(1.0), hlsl::promote<T>(1.0) / (orientedEta2 * orientedEta2 * (hlsl::promote<T>(554.33) - hlsl::promote<T>(380.7) * orientedEta)), TIR);
        T num = orientedEta * nbl::hlsl::mix<T>(hlsl::promote<T>(0.1921156102251088), orientedEta * hlsl::promote<T>(298.25) - hlsl::promote<T>(261.38) * orientedEta2 + hlsl::promote<T>(138.43), TIR);
        num = num + nbl::hlsl::mix<T>(hlsl::promote<T>(0.8078843897748912), hlsl::promote<T>(-1.67), TIR);
        return num * invdenum;
    }

    T orientedEta;
    T orientedEta2;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct OrientedEtaRcps
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static OrientedEtaRcps<T> create(scalar_type NdotI, T eta)
    {
        OrientedEtaRcps<T> retval;
        const bool backside = NdotI < scalar_type(0.0);
        const T rcpEta = hlsl::promote<T>(1.0) / eta;
        retval.value = backside ? eta : rcpEta;
        retval.value2 = retval.value * retval.value;
        return retval;
    }

    DiffuseCorrectionFactor<T> createDiffuseCorrectionFactor() NBL_CONST_MEMBER_FUNC
    {
        DiffuseCorrectionFactor<T> diffuseCorrectionFactor;
        diffuseCorrectionFactor.orientedEta = hlsl::promote<T>(1.0) / value;
        diffuseCorrectionFactor.orientedEta2 = hlsl::promote<T>(1.0) / value2;
        return diffuseCorrectionFactor;
    }

    T value;
    T value2;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct OrientedEtas
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static OrientedEtas<T> create(scalar_type NdotI, T eta)
    {
        OrientedEtas<T> retval;
        const bool backside = NdotI < scalar_type(0.0);
        const T rcpEta = hlsl::promote<T>(1.0) / eta;
        retval.value = backside ? rcpEta : eta;
        retval.rcp = backside ? eta : rcpEta;
        return retval;
    }

    OrientedEtaRcps<T> getReciprocals() NBL_CONST_MEMBER_FUNC
    {
        OrientedEtaRcps<T> retval;
        retval.value = rcp;
        retval.value2 = rcp * rcp;
        return retval;
    }

    T value;
    T rcp;
};

}


template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T>)
struct ComputeMicrofacetNormal
{
    using scalar_type = T;
    using vector_type = vector<scalar_type, 3>;
    using monochrome_type = vector<scalar_type, 1>;

    using unsigned_integer_type = typename unsigned_integer_of_size<sizeof(scalar_type)>::type;

    static ComputeMicrofacetNormal<T> create(NBL_CONST_REF_ARG(vector_type) V, NBL_CONST_REF_ARG(vector_type) L, NBL_CONST_REF_ARG(vector_type) H, scalar_type eta)
    {
        return create(V, L, hlsl::dot<vector_type>(V, H), eta);
    }

    static ComputeMicrofacetNormal<T> create(NBL_CONST_REF_ARG(vector_type) V, NBL_CONST_REF_ARG(vector_type) L, scalar_type VdotH, scalar_type eta)
    {
        ComputeMicrofacetNormal<T> retval;
        retval.V = V;
        retval.L = L;
        fresnel::OrientedEtas<monochrome_type> orientedEtas = fresnel::OrientedEtas<monochrome_type>::create(VdotH, eta);
        retval.orientedEta = orientedEtas.value;
        return retval;
    }

    // NDFs are defined in terms of `abs(NdotH)` and microfacets are two sided. Note that `N` itself is the definition of the upper hemisphere direction.
    // The possible directions of L form a cone around -V with the cosine of the angle equal higher or equal to min(orientedEta, 1.f/orientedEta), and vice versa.
    // This means that for:
    // - Eta>1  the L  will be longer than V projected on V, and VdotH<0 for all L
    // - whereas with Eta<1 the L is shorter, and VdotH>0 for all L
    // Because to be a refraction `VdotH` and `LdotH` must differ in sign, so whenever one is positive the other is negative.
    // Since we're considering single scattering, the V and L must enter the microfacet described by H same way they enter the macro-medium described by N.
    // All this means that by looking at the sign of VdotH we can also tell the sign of VdotN.
    // However the whole `V+L*eta` formula is backwards because what it should be is `-V-L*eta` so the sign flip is applied just to restore the H-finding to that value.

    // The math:
    // dot(V,H) = V2 + VdotL*eta = 1 + VdotL*eta, note that VdotL<=1 so VdotH>=0 when eta==1
    // then with valid transmission path constraint:
    // VdotH <= 1-orientedEta2 for orientedEta<1 -> VdotH<0
    // VdotH <= 0 for orientedEta>1
    // so for transmission VdotH<=0, H needs to be flipped to be consistent with oriented eta
    vector_type unnormalized(const bool _refract)
    {
        assert(hlsl::dot(V, L) <= -hlsl::min(orientedEta, scalar_type(1.0) / orientedEta));
        const scalar_type etaFactor = hlsl::mix(scalar_type(1.0), orientedEta.value, _refract);
        vector_type tmpH = V + L * etaFactor;
        tmpH = ieee754::flipSign<vector_type>(tmpH, _refract);
        return tmpH;
    }

    // returns normalized vector, but NaN when result is length 0
    vector_type normalized(const bool _refract)
    {
        const vector_type H = unnormalized(_refract);
        return hlsl::normalize<vector_type>(H);
    }

    // if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
    static bool isTransmissionPath(scalar_type NdotV, scalar_type NdotL)
    {
        return bool((hlsl::bit_cast<unsigned_integer_type>(NdotV) ^ hlsl::bit_cast<unsigned_integer_type>(NdotL)) & unsigned_integer_type(1u)<<(sizeof(scalar_type)*8u-1u));
    }

    static bool isValidMicrofacet(const bool transmitted, const scalar_type VdotL, const scalar_type NdotH, NBL_CONST_REF_ARG(fresnel::OrientedEtas<vector<scalar_type,1> >) orientedEta)
    {
        return !transmitted || (VdotL <= -hlsl::min(orientedEta.value, orientedEta.rcp) && NdotH >= nbl::hlsl::numeric_limits<scalar_type>::min);
    }

    vector_type V;
    vector_type L;
    scalar_type orientedEta;
};


template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct Reflect
{
    using this_t = Reflect<T>;
    using vector_type = vector<T, 3>;
    using scalar_type = T;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        return retval;
    }

    scalar_type getNdotI()
    {
        return hlsl::dot<vector_type>(N, I);
    }

    vector_type operator()()
    {
        return operator()(getNdotI());
    }

    vector_type operator()(const scalar_type NdotI)
    {
        return N * 2.0f * NdotI - I;
    }

    vector_type I;
    vector_type N;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct Refract
{
    using this_t = Refract<T>;
    using vector_type = vector<T, 3>;
    using monochrome_type = vector<T, 1>;
    using scalar_type = T;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        return retval;
    }

    scalar_type getNdotI()
    {
        return hlsl::dot<vector_type>(N, I);
    }

    scalar_type getNdotT(const scalar_type rcpOrientedEta2)
    {
        scalar_type NdotI = getNdotI();
        scalar_type NdotT2 = rcpOrientedEta2 * NdotI*NdotI + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        return ieee754::copySign(absNdotT, -NdotI);  // TODO: make a ieee754::copySignIntoPositive, see https://github.com/Devsh-Graphics-Programming/Nabla/pull/899#discussion_r2197473145
    }

    vector_type operator()(const scalar_type rcpOrientedEta)
    {
        return N * (getNdotI() * rcpOrientedEta + getNdotT(rcpOrientedEta*rcpOrientedEta)) - rcpOrientedEta * I;
    }

    vector_type operator()(const scalar_type rcpOrientedEta, const scalar_type NdotI, const scalar_type NdotT)
    {
        return N * (NdotI * rcpOrientedEta + NdotT) - rcpOrientedEta * I;
    }

    vector_type I;
    vector_type N;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::Scalar<T>)
struct ReflectRefract
{
    using this_t = ReflectRefract<T>;
    using vector_type = vector<T, 3>;
    using scalar_type = T;

    // when you know you'll reflect
    scalar_type getNdotR()
    {
        return refract.getNdotI();
    }

    // when you know you'll refract
    scalar_type getNdotT(const scalar_type rcpOrientedEta)
    {
        return refract.getNdotT(rcpOrientedEta*rcpOrientedEta);
    }

    scalar_type getNdotTorR(const bool doRefract, const scalar_type rcpOrientedEta)
    {
        return hlsl::mix(getNdotR(), getNdotT(rcpOrientedEta), doRefract);
    }

    vector_type operator()(const bool doRefract, const scalar_type rcpOrientedEta)
    {
        scalar_type NdotI = getNdotR();
        return refract.N * (NdotI * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract)) + hlsl::mix(NdotI, getNdotT(rcpOrientedEta), doRefract)) - refract.I * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract));
    }

    vector_type operator()(const scalar_type NdotTorR, const scalar_type rcpOrientedEta)
    {
        scalar_type NdotI = getNdotR();
        bool doRefract = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(NdotI, NdotTorR);
        return refract.N * (NdotI * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract)) + hlsl::mix(NdotI, NdotTorR, doRefract)) - refract.I * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract));
    }

    Refract<scalar_type> refract;
};


namespace fresnel
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Schlick
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Schlick<T> create(NBL_CONST_REF_ARG(T) F0, scalar_type clampedCosTheta)
    {
        Schlick<T> retval;
        retval.F0 = F0;
        retval.clampedCosTheta = clampedCosTheta;
        return retval;
    }

    T operator()()
    {
        assert(clampedCosTheta > scalar_type(0.0));
        assert(hlsl::all(hlsl::promote<T>(0.02) < F0 && F0 <= hlsl::promote<T>(1.0)));
        T x = 1.0 - clampedCosTheta;
        return F0 + (1.0 - F0) * x*x*x*x*x;
    }

    T F0;
    scalar_type clampedCosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Conductor
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Conductor<T> create(NBL_CONST_REF_ARG(T) eta, NBL_CONST_REF_ARG(T) etak, scalar_type clampedCosTheta)
    {
        Conductor<T> retval;
        retval.eta = eta;
        retval.etak2 = etak*etak;
        retval.clampedCosTheta = clampedCosTheta;
        return retval;
    }

    static Conductor<T> create(NBL_CONST_REF_ARG(complex_t<T>) eta, scalar_type clampedCosTheta)
    {
        Conductor<T> retval;
        retval.eta = eta.real();
        retval.etak2 = eta.imag()*eta.imag();
        retval.clampedCosTheta = clampedCosTheta;
        return retval;
    }

    T operator()()
    {
        const scalar_type cosTheta2 = clampedCosTheta * clampedCosTheta;
        //const float sinTheta2 = 1.0 - cosTheta2;

        const T etaLen2 = eta * eta + etak2;
        assert(hlsl::all(etaLen2 > hlsl::promote<T>(hlsl::exp2<scalar_type>(-numeric_limits<scalar_type>::digits))));
        const T etaCosTwice = eta * clampedCosTheta * 2.0f;

        const T rs_common = etaLen2 + (T)(cosTheta2);
        const T rs2 = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);

        const T rp_common = etaLen2 * cosTheta2 + (T)(1.0);
        const T rp2 = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);

        return (rs2 + rp2) * 0.5f;
    }

    T eta;
    T etak2;
    scalar_type clampedCosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Dielectric
{
    using scalar_type = typename vector_traits<T>::scalar_type;

    static Dielectric<T> create(NBL_CONST_REF_ARG(T) eta, scalar_type cosTheta)
    {
        Dielectric<T> retval;
        scalar_type absCosTheta = hlsl::abs(cosTheta);
        OrientedEtas<T> orientedEta = OrientedEtas<T>::create(absCosTheta, eta);
        retval.orientedEta2 = orientedEta.value * orientedEta.value;
        retval.absCosTheta = absCosTheta;
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
        return __call(orientedEta2, absCosTheta);
    }

    T orientedEta2;
    scalar_type absCosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
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
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<T> || concepts::FloatingPointLikeVectorial<T>)
T thinDielectricInfiniteScatter(const T singleInterfaceReflectance)
{
    const T doubleInterfaceReflectance = singleInterfaceReflectance * singleInterfaceReflectance;
    return hlsl::mix<T>(hlsl::promote<T>(1.0), (singleInterfaceReflectance - doubleInterfaceReflectance) / (hlsl::promote<T>(1.0) - doubleInterfaceReflectance) * 2.0f, doubleInterfaceReflectance > hlsl::promote<T>(0.9999));
}

}

}
}
}

#endif
