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
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/colorspace.hlsl"
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
        retval.value = hlsl::mix(rcpEta, eta, backside);
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
        retval.value = hlsl::mix(eta, rcpEta, backside);
        retval.rcp = hlsl::mix(rcpEta, eta, backside);
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
        const scalar_type etaFactor = hlsl::mix(scalar_type(1.0), orientedEta, _refract);
        vector_type tmpH = V + L * etaFactor;
        tmpH = ieee754::flipSign<vector_type>(tmpH, _refract && orientedEta > scalar_type(1.0));
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
        return !transmitted || (VdotL <= -hlsl::min(orientedEta.value[0], orientedEta.rcp[0]) && NdotH >= nbl::hlsl::numeric_limits<scalar_type>::min);
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

    scalar_type getNdotI() NBL_CONST_MEMBER_FUNC
    {
        return hlsl::dot<vector_type>(N, I);
    }

    vector_type operator()() NBL_CONST_MEMBER_FUNC
    {
        return operator()(getNdotI());
    }

    vector_type operator()(const scalar_type NdotI) NBL_CONST_MEMBER_FUNC
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
    using scalar_type = T;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        return retval;
    }

    scalar_type getNdotI() NBL_CONST_MEMBER_FUNC
    {
        return hlsl::dot<vector_type>(N, I);
    }

    scalar_type getNdotT(const scalar_type rcpOrientedEta2) NBL_CONST_MEMBER_FUNC
    {
        scalar_type NdotI = getNdotI();
        scalar_type NdotT2 = rcpOrientedEta2 * NdotI*NdotI + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        return ieee754::copySign(absNdotT, -NdotI);  // TODO: make a ieee754::copySignIntoPositive, see https://github.com/Devsh-Graphics-Programming/Nabla/pull/899#discussion_r2197473145
    }

    // refraction takes eta as ior_incoming/ior_transmitted due to snell's law
    vector_type operator()(const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        return N * (getNdotI() * rcpOrientedEta + getNdotT(rcpOrientedEta*rcpOrientedEta)) - rcpOrientedEta * I;
    }

    vector_type operator()(const scalar_type rcpOrientedEta, const scalar_type NdotI, const scalar_type NdotT) NBL_CONST_MEMBER_FUNC
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
    scalar_type getNdotR() NBL_CONST_MEMBER_FUNC
    {
        return refract.getNdotI();
    }

    // when you know you'll refract
    scalar_type getNdotT(const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        return refract.getNdotT(rcpOrientedEta*rcpOrientedEta);
    }

    scalar_type getNdotTorR(const bool doRefract, const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        return hlsl::mix(getNdotR(), getNdotT(rcpOrientedEta), doRefract);
    }

    vector_type operator()(const bool doRefract, const scalar_type rcpOrientedEta, NBL_REF_ARG(scalar_type) out_IdotTorR) NBL_CONST_MEMBER_FUNC
    {
        scalar_type NdotI = getNdotR();
        const scalar_type a = hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract);
        const scalar_type b = NdotI * a + getNdotTorR(doRefract, rcpOrientedEta);
        // assuming `I` is normalized
        out_IdotTorR = NdotI * b - a;
        return refract.N * b - refract.I * a;
    }

    vector_type operator()(const bool doRefract, const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        scalar_type dummy;
        return operator()(doRefract, rcpOrientedEta, dummy);
    }

    vector_type operator()(const scalar_type NdotTorR, const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        scalar_type NdotI = getNdotR();
        bool doRefract = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(NdotI, NdotTorR);
        return refract.N * (NdotI * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract)) + hlsl::mix(NdotI, NdotTorR, doRefract)) - refract.I * (hlsl::mix<scalar_type>(1.0f, rcpOrientedEta, doRefract));
    }

    Refract<scalar_type> refract;
};


namespace fresnel
{

#define NBL_CONCEPT_NAME Fresnel
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (fresnel, T)
#define NBL_CONCEPT_PARAM_1 (cosTheta, typename T::scalar_type)
NBL_CONCEPT_BEGIN(2)
#define fresnel NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define cosTheta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::eta_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((fresnel(cosTheta)), ::nbl::hlsl::is_same_v, typename T::vector_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((fresnel.getOrientedEtaRcps()), ::nbl::hlsl::is_same_v, OrientedEtaRcps<typename T::eta_type>))
);
#undef cosTheta
#undef fresnel
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME TwoSidedFresnel
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (fresnel, T)
#define NBL_CONCEPT_PARAM_1 (cosTheta, typename T::scalar_type)
NBL_CONCEPT_BEGIN(2)
#define fresnel NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define cosTheta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Fresnel, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((fresnel.getRefractionOrientedEta()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((fresnel.getReorientedFresnel(cosTheta)), ::nbl::hlsl::is_same_v, T))
);
#undef cosTheta
#undef fresnel
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Schlick
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;
    using eta_type = vector_type;

    NBL_CONSTEXPR_STATIC_INLINE bool ReturnsMonochrome = vector_traits<vector_type>::Dimension == 1;

    static Schlick<T> create(NBL_CONST_REF_ARG(T) F0)
    {
        Schlick<T> retval;
        retval.F0 = F0;
        return retval;
    }

    T operator()(const scalar_type clampedCosTheta) NBL_CONST_MEMBER_FUNC
    {
        assert(clampedCosTheta >= scalar_type(0.0));
        assert(hlsl::all(hlsl::promote<T>(0.02) < F0 && F0 <= hlsl::promote<T>(1.0)));
        T x = 1.0 - clampedCosTheta;
        return F0 + (1.0 - F0) * x*x*x*x*x;
    }

    OrientedEtaRcps<eta_type> getOrientedEtaRcps() NBL_CONST_MEMBER_FUNC
    {
        const eta_type sqrtF0 = hlsl::sqrt(F0);        
        OrientedEtaRcps<eta_type> rcpEta;
        rcpEta.value = (hlsl::promote<eta_type>(1.0) - sqrtF0) / (hlsl::promote<eta_type>(1.0) + sqrtF0);
        rcpEta.value2 = rcpEta.value * rcpEta.value;
        return rcpEta;
    }

    T F0;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Conductor
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;
    using eta_type = vector_type;

    NBL_CONSTEXPR_STATIC_INLINE bool ReturnsMonochrome = vector_traits<vector_type>::Dimension == 1;

    static Conductor<T> create(NBL_CONST_REF_ARG(T) eta, NBL_CONST_REF_ARG(T) etak)
    {
        Conductor<T> retval;
        retval.eta = eta;
        retval.etak2 = etak * etak;
        retval.etaLen2 = eta * eta + retval.etak2;
        return retval;
    }

    static Conductor<T> create(NBL_CONST_REF_ARG(complex_t<T>) eta)
    {
        Conductor<T> retval;
        retval.eta = eta.real();
        retval.etak2 = eta.imag() * eta.imag();
        retval.etaLen2 = eta * eta + retval.etak2;
        return retval;
    }

    static void __polarized(const T orientedEta, const T etaLen2, const T clampedCosTheta, NBL_REF_ARG(T) Rp, NBL_REF_ARG(T) Rs)
    {
        const T cosTheta_2 = clampedCosTheta * clampedCosTheta;
        const T eta = orientedEta;

        assert(hlsl::all(etaLen2 > hlsl::promote<T>(hlsl::exp2<scalar_type>(-numeric_limits<scalar_type>::digits))));
        T t1 = etaLen2 * cosTheta_2;
        const T etaCosTwice = eta * clampedCosTheta * hlsl::promote<T>(2.0);

        const T rs_common = etaLen2 + cosTheta_2;
        Rs = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);
        const T rp_common = t1 + hlsl::promote<T>(1.0);
        Rp = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);
    }

    T operator()(const scalar_type clampedCosTheta) NBL_CONST_MEMBER_FUNC
    {
        T rs2, rp2;
        __polarized(eta, etaLen2, hlsl::promote<T>(clampedCosTheta), rp2, rs2);

        return (rs2 + rp2) * hlsl::promote<T>(0.5);
    }

    OrientedEtaRcps<eta_type> getOrientedEtaRcps() NBL_CONST_MEMBER_FUNC
    {
        OrientedEtaRcps<eta_type> rcpEta;
        rcpEta.value = hlsl::promote<eta_type>(1.0) / eta;
        rcpEta.value2 = rcpEta.value * rcpEta.value;
        return rcpEta;
    }

    T eta;
    T etak2;
    T etaLen2;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Dielectric
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;
    using eta_type = vector_type;

    NBL_CONSTEXPR_STATIC_INLINE bool ReturnsMonochrome = vector_traits<vector_type>::Dimension == 1;

    static Dielectric<T> create(NBL_CONST_REF_ARG(OrientedEtas<T>) orientedEta)
    {
        Dielectric<T> retval;
        retval.orientedEta = orientedEta;
        retval.orientedEta2 = orientedEta.value * orientedEta.value;
        return retval;
    }

    static void __polarized(const T orientedEta2, const T cosTheta, NBL_REF_ARG(T) Rp, NBL_REF_ARG(T) Rs)
    {
        const T sinTheta2 = hlsl::promote<T>(1.0) - cosTheta * cosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        T t0 = hlsl::sqrt(hlsl::max(orientedEta2 - sinTheta2, hlsl::promote<T>(0.0)));
        T t2 = orientedEta2 * cosTheta;

        T rp = (t0 - t2) / (t0 + t2);
        Rp = rp * rp;
        T rs = (cosTheta - t0) / (cosTheta + t0);
        Rs = rs * rs;
    }

    static T __call(NBL_CONST_REF_ARG(T) orientedEta2, const scalar_type clampedCosTheta)
    {
        T rs2, rp2;
        __polarized(orientedEta2, hlsl::promote<T>(clampedCosTheta), rp2, rs2);

        return (rs2 + rp2) * hlsl::promote<T>(0.5);
    }

    T operator()(const scalar_type clampedCosTheta) NBL_CONST_MEMBER_FUNC
    {
        return __call(orientedEta2, clampedCosTheta);
    }

    // default to monochrome, but it is possible to have RGB fresnel without dispersion fixing the refraction Eta
    // to be something else than the etas used to compute RGB reflectance or some sort of interpolation of them
    scalar_type getRefractionOrientedEta() NBL_CONST_MEMBER_FUNC { return orientedEta.value[0]; }
    OrientedEtaRcps<T> getOrientedEtaRcps() NBL_CONST_MEMBER_FUNC { return orientedEta.getReciprocals(); }

    Dielectric<T> getReorientedFresnel(const scalar_type NdotI) NBL_CONST_MEMBER_FUNC
    {
        OrientedEtas<T> eta = OrientedEtas<T>::create(NdotI, orientedEta.value);
        return Dielectric<T>::create(eta);
    }

    OrientedEtas<T> orientedEta;
    T orientedEta2;
};

// adapted from https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
template<typename T, bool SupportsTransmission, typename Colorspace = colorspace::scRGB NBL_STRUCT_CONSTRAINABLE>
struct Iridescent;

namespace impl
{
template<typename T, bool SupportsTransmission NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct iridescent_helper
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;

    // returns phi, the phase shift for each plane of polarization (p,s)
    static void phase_shift(const vector_type ior1, const vector_type ior2, const vector_type iork2, const vector_type cosTheta, NBL_REF_ARG(vector_type) phiS, NBL_REF_ARG(vector_type) phiP)
    {
        const vector_type cosTheta2 = cosTheta * cosTheta;
        const vector_type sinTheta2 = hlsl::promote<vector_type>(1.0) - cosTheta2;
        const vector_type ior1_2 = ior1*ior1;
        const vector_type ior2_2 = ior2*ior2;
        const vector_type iork2_2 = iork2*iork2;

        const vector_type z = ior2_2 * (hlsl::promote<vector_type>(1.0) - iork2_2) - ior1_2 * sinTheta2;
        const vector_type w = hlsl::sqrt(z*z + scalar_type(4.0) * ior2_2 * ior2_2 * iork2_2);
        const vector_type a2 = hlsl::max(z + w, hlsl::promote<vector_type>(0.0)) * hlsl::promote<vector_type>(0.5);
        const vector_type b2 = hlsl::max(w - z, hlsl::promote<vector_type>(0.0)) * hlsl::promote<vector_type>(0.5);
        const vector_type a = hlsl::sqrt(a2);
        const vector_type b = hlsl::sqrt(b2);

        phiS = hlsl::atan2(scalar_type(2.0) * ior1 * b * cosTheta, a2 + b2 - ior1_2*cosTheta2);
        const vector_type k2_plus_one = hlsl::promote<vector_type>(1.0) + iork2_2;
        phiP = hlsl::atan2(scalar_type(2.0) * ior1 * ior2_2 * cosTheta * (scalar_type(2.0) * iork2 * a - (hlsl::promote<vector_type>(1.0) - iork2_2) * b),
                ior2_2 * cosTheta2 * k2_plus_one * k2_plus_one - ior1_2*(a2+b2));
    }

    // Evaluation XYZ sensitivity curves in Fourier space
    static vector_type evalSensitivity(vector_type opd, vector_type shift)
    {
        // Use Gaussian fits, given by 3 parameters: val, pos and var
        vector_type phase = scalar_type(2.0) * numbers::pi<scalar_type> * opd * scalar_type(1.0e-9);
        vector_type phase2 = phase * phase;
        vector_type val = vector_type(5.4856e-13, 4.4201e-13, 5.2481e-13);
        vector_type pos = vector_type(1.6810e+06, 1.7953e+06, 2.2084e+06);
        vector_type var = vector_type(4.3278e+09, 9.3046e+09, 6.6121e+09);
        vector_type xyz = val * hlsl::sqrt(scalar_type(2.0) * numbers::pi<scalar_type> * var) * hlsl::cos(pos * phase + shift) * hlsl::exp(-var * phase2);
        xyz.x = xyz.x + scalar_type(9.7470e-14) * hlsl::sqrt(scalar_type(2.0) * numbers::pi<scalar_type> * scalar_type(4.5282e+09)) * hlsl::cos(scalar_type(2.2399e+06) * phase[0] + shift[0]) * hlsl::exp(scalar_type(-4.5282e+09) * phase2[0]);
        return xyz / scalar_type(1.0685e-7);
    }

    template<typename Colorspace>
    static T __call(const vector_type _D, const vector_type ior1, const vector_type ior2, const vector_type ior3, const vector_type iork3,
                    const vector_type eta12, const vector_type eta23, const vector_type etak23, const scalar_type clampedCosTheta)
    {
        const vector_type wavelengths = vector_type(Colorspace::wavelength_R, Colorspace::wavelength_G, Colorspace::wavelength_B);

        const scalar_type cosTheta_1 = clampedCosTheta;
        vector_type R12p, R23p, R12s, R23s;
        vector_type cosTheta_2;
        vector<bool,vector_traits<vector_type>::Dimension> notTIR;
        {
            const vector_type scale = scalar_type(1.0)/eta12;
            const vector_type cosTheta2_2 = hlsl::promote<vector_type>(1.0) - hlsl::promote<vector_type>(scalar_type(1.0)-cosTheta_1*cosTheta_1) * scale * scale;
            notTIR = cosTheta2_2 > hlsl::promote<vector_type>(0.0);
            cosTheta_2 = hlsl::sqrt(hlsl::max(cosTheta2_2, hlsl::promote<vector_type>(0.0)));
        }

        if (hlsl::any(notTIR))
        {
            Dielectric<vector_type>::__polarized(eta12 * eta12, hlsl::promote<vector_type>(cosTheta_1), R12p, R12s);

            // Reflected part by the base
            // if kappa==0, base material is dielectric
            NBL_IF_CONSTEXPR(SupportsTransmission)
                Dielectric<vector_type>::__polarized(eta23 * eta23, cosTheta_2, R23p, R23s);
            else
            {
                vector_type etaLen2 = eta23 * eta23 + etak23 * etak23;
                Conductor<vector_type>::__polarized(eta23, etaLen2, cosTheta_2, R23p, R23s);
            }
        }

        // Check for total internal reflection
        const vector_type notTIRFactor = vector_type(notTIR); // 0 when TIR, 1 otherwise
        R12s = R12s * notTIRFactor;
        R12p = R12p * notTIRFactor;
        R23s = R23s * notTIRFactor;
        R23p = R23p * notTIRFactor;

        // Compute the transmission coefficients
        vector_type T121p = hlsl::promote<vector_type>(1.0) - R12p;
        vector_type T121s = hlsl::promote<vector_type>(1.0) - R12s;

        // Optical Path Difference
        const vector_type D = _D * cosTheta_2;
        const vector_type Dphi = hlsl::promote<vector_type>(2.0 * numbers::pi<scalar_type>) * D / wavelengths;

        vector_type phi21p, phi21s, phi23p, phi23s, r123s, r123p, Rs;
        vector_type I = hlsl::promote<vector_type>(0.0);

        // Evaluate the phase shift
        phase_shift(ior1, ior2, hlsl::promote<vector_type>(0.0), hlsl::promote<vector_type>(cosTheta_1), phi21s, phi21p);
        phase_shift(ior2, ior3, iork3, cosTheta_2, phi23s, phi23p);
        phi21p = hlsl::promote<vector_type>(numbers::pi<scalar_type>) - phi21p;
        phi21s = hlsl::promote<vector_type>(numbers::pi<scalar_type>) - phi21s;

        r123p = hlsl::sqrt(R12p*R23p);
        r123s = hlsl::sqrt(R12s*R23s);

        vector_type C0, Cm, Sm;
        const vector_type S0 = hlsl::promote<vector_type>(1.0);

        // Iridescence term using spectral antialiasing
        // Reflectance term for m=0 (DC term amplitude)
        Rs = (T121p*T121p*R23p) / (hlsl::promote<vector_type>(1.0) - R12p*R23p);
        C0 = R12p + Rs;
        I += C0 * S0;

        // Reflectance term for m>0 (pairs of diracs)
        Cm = Rs - T121p;
        NBL_UNROLL for (int m=1; m<=2; ++m)
        {
            Cm *= r123p;
            Sm  = hlsl::promote<vector_type>(2.0) * evalSensitivity(hlsl::promote<vector_type>(scalar_type(m))*D, hlsl::promote<vector_type>(scalar_type(m))*(phi23p+phi21p));
            I  += Cm*Sm;
        }

        // Reflectance term for m=0 (DC term amplitude)
        Rs = (T121s*T121s*R23s) / (hlsl::promote<vector_type>(1.0) - R12s*R23s);
        C0 = R12s + Rs;
        I += C0 * S0;

        // Reflectance term for m>0 (pairs of diracs)
        Cm = Rs - T121s;
        NBL_UNROLL for (int m=1; m<=2; ++m)
        {
            Cm *= r123s;
            Sm  = hlsl::promote<vector_type>(2.0) * evalSensitivity(hlsl::promote<vector_type>(scalar_type(m))*D, hlsl::promote<vector_type>(scalar_type(m)) *(phi23s+phi21s));
            I  += Cm*Sm;
        }

        return hlsl::max(colorspace::scRGB::FromXYZ(I) * hlsl::promote<vector_type>(0.5), hlsl::promote<vector_type>(0.0));
    }
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)    
struct iridescent_base
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;

    vector_type D;
    vector_type ior1;
    vector_type ior2;
    vector_type ior3;
    vector_type iork3;
    vector_type eta12;      // outside (usually air 1.0) -> thin-film IOR
    vector_type eta23;      // thin-film -> base material IOR
};
}

template<typename T, typename Colorspace>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct Iridescent<T, false, Colorspace NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) > : impl::iridescent_base<T>
{
    using this_t = Iridescent<T,false,Colorspace>;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;  // assert dim==3?
    using eta_type = vector_type;
    using base_type = impl::iridescent_base<T>;

    NBL_CONSTEXPR_STATIC_INLINE bool ReturnsMonochrome = vector_traits<vector_type>::Dimension == 1;

    struct SCreationParams
    {
        scalar_type Dinc;   // thickness of thin film in nanometers, rec. 100-25000nm
        vector_type ior1;   // outside (usually air 1.0)
        vector_type ior2;   // thin-film ior
        vector_type ior3;   // base mat ior
        vector_type iork3;
    };
    using creation_params_type = SCreationParams;

    static this_t create(NBL_CONST_REF_ARG(creation_params_type) params)
    {
        this_t retval;
        retval.D = hlsl::promote<vector_type>(2.0 * params.Dinc) * params.ior2;
        retval.ior1 = params.ior1;
        retval.ior2 = params.ior2;
        retval.ior3 = params.ior3;
        retval.iork3 = params.iork3;
        retval.eta12 = params.ior2/params.ior1;
        retval.eta23 = params.ior3/params.ior2;
        retval.etak23 = params.iork3/params.ior2;
        return retval;
    }

    T operator()(const scalar_type clampedCosTheta) NBL_CONST_MEMBER_FUNC
    {
        return impl::iridescent_helper<T,false>::template __call<Colorspace>(base_type::D, base_type::ior1, base_type::ior2, base_type::ior3, base_type::iork3,
                                                            base_type::eta12, base_type::eta23, getEtak23(), clampedCosTheta);
    }

    OrientedEtaRcps<eta_type> getOrientedEtaRcps() NBL_CONST_MEMBER_FUNC
    {
        OrientedEtaRcps<eta_type> rcpEta;
        rcpEta.value = hlsl::promote<eta_type>(1.0) / base_type::eta23;
        rcpEta.value2 = rcpEta.value * rcpEta.value;
        return rcpEta;
    }

    vector_type getEtak23() NBL_CONST_MEMBER_FUNC
    {
        return etak23;
    }

    vector_type etak23;     // thin-film -> complex component
};

template<typename T, typename Colorspace>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<T>)
struct Iridescent<T, true, Colorspace NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<T>) > : impl::iridescent_base<T>
{
    using this_t = Iridescent<T,true,Colorspace>;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;  // assert dim==3?
    using eta_type = vector<scalar_type, 1>;
    using base_type = impl::iridescent_base<T>;

    NBL_CONSTEXPR_STATIC_INLINE bool ReturnsMonochrome = vector_traits<vector_type>::Dimension == 1;

    struct SCreationParams
    {
        scalar_type Dinc;   // thickness of thin film in nanometers, rec. 100-25000nm
        vector_type ior1;   // outside (usually air 1.0)
        vector_type ior2;   // thin-film ior
        vector_type ior3;   // base mat ior
    };
    using creation_params_type = SCreationParams;

    static this_t create(NBL_CONST_REF_ARG(creation_params_type) params)
    {
        this_t retval;
        retval.D = hlsl::promote<vector_type>(2.0 * params.Dinc) * params.ior2;
        retval.ior1 = params.ior1;
        retval.ior2 = params.ior2;
        retval.ior3 = params.ior3;
        retval.eta12 = params.ior2/params.ior1;
        retval.eta23 = params.ior3/params.ior2;
        return retval;
    }

    T operator()(const scalar_type clampedCosTheta) NBL_CONST_MEMBER_FUNC
    {
        return impl::iridescent_helper<T,true>::template __call<Colorspace>(base_type::D, base_type::ior1, base_type::ior2, base_type::ior3, getEtak23(),
                                                            base_type::eta12, base_type::eta23, getEtak23(), clampedCosTheta);
    }

    scalar_type getRefractionOrientedEta() NBL_CONST_MEMBER_FUNC { return base_type::ior3[0] / base_type::ior1[0]; }
    OrientedEtaRcps<eta_type> getOrientedEtaRcps() NBL_CONST_MEMBER_FUNC
    {
        OrientedEtaRcps<eta_type> rcpEta;
        rcpEta.value = base_type::ior1[0] / base_type::ior3[0];
        rcpEta.value2 = rcpEta.value * rcpEta.value;
        return rcpEta;
    }

    this_t getReorientedFresnel(const scalar_type NdotI) NBL_CONST_MEMBER_FUNC
    {
        const bool flip = NdotI < scalar_type(0.0);
        this_t orientedFresnel;
        orientedFresnel.D = base_type::D;
        orientedFresnel.ior1 = hlsl::mix(base_type::ior1, base_type::ior3, flip);
        orientedFresnel.ior2 = base_type::ior2;
        orientedFresnel.ior3 = hlsl::mix(base_type::ior3, base_type::ior1, flip);
        orientedFresnel.eta12 = hlsl::mix(base_type::eta12, hlsl::promote<vector_type>(1.0)/base_type::eta23, flip);
        orientedFresnel.eta23 = hlsl::mix(base_type::eta23, hlsl::promote<vector_type>(1.0)/base_type::eta12, flip);
        return orientedFresnel;
    }

    vector_type getEtak23() NBL_CONST_MEMBER_FUNC
    {
        return hlsl::promote<vector_type>(0.0);
    }
};


// gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<T> || concepts::FloatingPointLikeVectorial<T>)
T thinDielectricInfiniteScatter(const T singleInterfaceReflectance)
{
    const T doubleInterfaceReflectance = singleInterfaceReflectance * singleInterfaceReflectance;
    return hlsl::mix<T>((singleInterfaceReflectance - doubleInterfaceReflectance) / (hlsl::promote<T>(1.0) - doubleInterfaceReflectance) * 2.0f, hlsl::promote<T>(1.0), doubleInterfaceReflectance > hlsl::promote<T>(0.9999));
}

}

}
}
}

#endif
