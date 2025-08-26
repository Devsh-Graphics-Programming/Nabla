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
#include "nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl"
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
NBL_CONCEPT_BEGIN(1)
#define fresnel NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((fresnel()), ::nbl::hlsl::is_same_v, typename T::vector_type))
);
#undef fresnel
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Schlick
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;

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
    using vector_type = T;

    static Conductor<T> create(NBL_CONST_REF_ARG(T) eta, NBL_CONST_REF_ARG(T) etak, scalar_type clampedCosTheta)
    {
        Conductor<T> retval;
        retval.eta = eta;
        retval.etak = etak;
        retval.etak2 = etak*etak;
        retval.clampedCosTheta = clampedCosTheta;
        return retval;
    }

    static Conductor<T> create(NBL_CONST_REF_ARG(complex_t<T>) eta, scalar_type clampedCosTheta)
    {
        Conductor<T> retval;
        retval.eta = eta.real();
        retval.etak = eta.imag();
        retval.etak2 = eta.imag()*eta.imag();
        retval.clampedCosTheta = clampedCosTheta;
        return retval;
    }

    // returns reflectance R = (rp, rs), phi is the phase shift for each plane of polarization (p,s)
    static vector<scalar_type, 2> __polarized_w_phase_shift(NBL_CONST_REF_ARG(T) orientedEta, scalar_type orientedEtak, scalar_type cosTheta, NBL_REF_ARG(vector<scalar_type, 2>) phi)
    {
        scalar_type cosTheta_2 = cosTheta * cosTheta;
        scalar_type sinTheta2 = scalar_type(1.0) - cosTheta_2;
        const scalar_type eta = orientedEta[0];
        const scalar_type eta2 = eta*eta;
        const scalar_type etak = orientedEtak;
        const scalar_type etak2 = etak*etak;

        scalar_type z = eta2 - etak2 - sinTheta2;
        scalar_type w = hlsl::sqrt(z * z + scalar_type(4.0) * eta2 * eta2 * etak2);
        scalar_type a2 = (z + w) / scalar_type(2.0);
        scalar_type b2 = (w - z) / scalar_type(2.0);
        scalar_type b = hlsl::sqrt(b2);

        const scalar_type etaLen2 = eta2 + etak2;
        assert(etaLen2 > hlsl::exp2<scalar_type>(-numeric_limits<scalar_type>::digits));
        scalar_type t1 = etaLen2 * cosTheta_2;
        const scalar_type etaCosTwice = eta * clampedCosTheta * scalar_type(2.0);

        phi.y = hlsl::atan(scalar_type(2.0) * b * cosTheta, a2 + b2 - cosTheta_2) + numbers::pi<T>;
        phi.x = hlsl::atan(scalar_type(2.0) * eta2 * cosTheta * (scalar_type(2.0) * etak * hlsl::sqrt(a2) - etak2 * b), t1 - a2 + b2);

        vector<scalar_type, 2> R;   // (rp, rs)
        const T rs_common = etaLen2 + cosTheta_2;
        R.y = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);
        const T rp_common = t1 + scalar_type(1.0);
        R.x = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);
        return R;
    }

    T operator()()
    {
        const scalar_type cosTheta_2 = clampedCosTheta * clampedCosTheta;
        //const float sinTheta2 = 1.0 - cosTheta_2;

        const T etaLen2 = eta * eta + etak2;
        assert(hlsl::all(etaLen2 > hlsl::promote<T>(hlsl::exp2<scalar_type>(-numeric_limits<scalar_type>::digits))));
        const T etaCosTwice = eta * clampedCosTheta * hlsl::promote<T>(2.0);

        const T rs_common = etaLen2 + hlsl::promote<T>(cosTheta_2);
        const T rs2 = (rs_common - etaCosTwice) / (rs_common + etaCosTwice);

        const T rp_common = etaLen2 * cosTheta_2 + hlsl::promote<T>(1.0);
        const T rp2 = (rp_common - etaCosTwice) / (rp_common + etaCosTwice);

        return (rs2 + rp2) * hlsl::promote<T>(0.5);
    }

    T eta;
    T etak;
    T etak2;
    scalar_type clampedCosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Dielectric
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;

    static Dielectric<T> create(NBL_CONST_REF_ARG(T) eta, scalar_type cosTheta)
    {
        Dielectric<T> retval;
        scalar_type absCosTheta = hlsl::abs(cosTheta);
        retval.orientedEta = OrientedEtas<T>::create(absCosTheta, eta);
        retval.absCosTheta = absCosTheta;
        return retval;
    }

    // returns reflectance R = (rp, rs), phi is the phase shift for each plane of polarization (p,s)
    static vector<scalar_type, 2> __polarized_w_phase_shift(NBL_CONST_REF_ARG(T) orientedEta, scalar_type cosTheta, NBL_REF_ARG(vector<scalar_type, 2>) phi)
    {
        scalar_type sinTheta2 = scalar_type(1.0) - cosTheta * cosTheta;
        const scalar_type eta = orientedEta[0];
        const scalar_type eta2 = eta * eta;

        // TIR
        if (eta2 * sinTheta2 > scalar_type(1.0))
        {
            const scalar_type phase_advance_s = hlsl::sqrt(sinTheta2 - scalar_type(1.0) / eta2) / cosTheta;
            phi = scalar_type(2.0) * hlsl::atan(vector<scalar_type, 2>(-eta2 * phase_advance_s,
                                                                        phase_advance_s));
            return hlsl::promote<vector<scalar_type, 2> >(1.0);
        }

        scalar_type t0 = hlsl::sqrt(eta2 - sinTheta2);
        scalar_type t2 = eta * cosTheta;

        vector<scalar_type, 2> R;   // (rp, rs)
        R.x = (t0 - t2) / (t0 + t2);
        R.y = (cosTheta - t0) / (cosTheta + t0);
        phi = hlsl::mix(hlsl::promote<vector<scalar_type, 2> >(0.0), hlsl::promote<vector<scalar_type, 2> >(numbers::pi<T>), R < vector<scalar_type, 2>(0.0));
        return R;
    }

    static T __call(NBL_CONST_REF_ARG(T) orientedEta2, scalar_type absCosTheta)
    {
        const scalar_type sinTheta2 = scalar_type(1.0) - absCosTheta * absCosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        const T t0 = hlsl::sqrt<T>(hlsl::max<T>(orientedEta2 - sinTheta2, hlsl::promote<T>(0.0)));
        const T rs = (hlsl::promote<T>(absCosTheta) - t0) / (hlsl::promote<T>(absCosTheta) + t0);

        const T t2 = orientedEta2 * absCosTheta;
        const T rp = (t0 - t2) / (t0 + t2);

        return (rs * rs + rp * rp) * scalar_type(0.5);
    }

    T operator()()
    {
        return __call(orientedEta.value * orientedEta.value, absCosTheta);
    }

    OrientedEtas<T> orientedEta;
    scalar_type absCosTheta;
};

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct DielectricFrontFaceOnly
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector_type = T;

    static DielectricFrontFaceOnly<T> create(NBL_CONST_REF_ARG(T) eta, scalar_type absCosTheta)
    {
        Dielectric<T> retval;
        retval.orientedEta = OrientedEtas<T>::create(absCosTheta, eta);
        retval.absCosTheta = hlsl::abs<T>(absCosTheta);
        return retval;
    }

    T operator()()
    {
        return Dielectric<T>::__call(orientedEta.value * orientedEta.value, absCosTheta);
    }

    OrientedEtas<T> orientedEta;
    scalar_type absCosTheta;
};

// adapted from https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct Iridescent
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector_type = T;

    // Depolarization functions for natural light
    static scalar_type depolarize(vector2_type polV)
    {
        return scalar_type(0.5) * (polV.x + polV.y);
    }
    static vector_type depolarizeColor(vector_type colS, vector_type colP)
    {
        return scalar_type(0.5) * (colS + colP);
    }

    // Evaluation XYZ sensitivity curves in Fourier space
    static vector_type evalSensitivity(scalar_type opd, scalar_type shift)
    {
        // Use Gaussian fits, given by 3 parameters: val, pos and var
        scalar_type phase = scalar_type(2.0) * numbers::pi<T> * opd * scalar_type(1.0e-6);
        vector_type val = vector_type(5.4856e-13, 4.4201e-13, 5.2481e-13);
        vector_type pos = vector_type(1.6810e+06, 1.7953e+06, 2.2084e+06);
        vector_type var = vector_type(4.3278e+09, 9.3046e+09, 6.6121e+09);
        vector_type xyz = val * hlsl::sqrt(scalar_type(2.0) * numbers::pi<T> * var) * hlsl::cos(pos * phase + shift) * hlsl::exp(- var * phase * phase);
        xyz.x += scalar_type(9.7470e-14) * hlsl::sqrt(scalar_type(2.0) * numbers::pi<T> * scalar_type(4.5282e+09)) * hlsl::cos(scalar_type(2.2399e+06) * phase + shift) * hlsl::exp(scalar_type(-4.5282e+09) * phase * phase);
        return xyz / scalar_type(1.0685e-7);
    }

    T operator()()
    {
        // Force ior_1 -> 1.0 when Dinc -> 0.0
        scalar_type ior_1 = hlsl::mix(1.0, ior1, hlsl::smoothstep(0.0, 0.03, Dinc));

        scalar_type eta1 = ior_1;   // ior/1.0 (air)
        scalar_type rcp_eta1 = 1.0/eta1;
        scalar_type cosTheta_1 = cosTheta;
        scalar_type cosTheta_2 = hlsl::sqrt(1.0 - rcp_eta1*rcp_eta1*(1.0-cosTheta_1*cosTheta_1) );

        // First interface
        vector2_type phi12;
        vector2_type R12 = Dielectric<vector<scalar_type, 1> >::__polarized_w_phase_shift(hlsl::promote<vector<scalar_type, 1> >(eta1), cosTheta_1, phi12);
        vector2_type R21 = R12;
        vector2_type T121 = hlsl::promote<vector2_type>(1.0) - R12;
        vector2_type phi21 = hlsl::promote<vector2_type>(numbers::pi<T>) - phi12;

        // Second interface
        scalar_type eta2 = ior2/ior_1;
        scalar_type etak2 = iork2/ior_1;
        vector2_type R23, phi23;
        Conductor<vector<scalar_type, 1> >::__polarized_w_phase_shift(hlsl::promote<vector<scalar_type, 1> >(eta2), etak2, cosTheta_2, phi23);

        // Phase shift
        scalar_type OPD = Dinc*cosTheta_2;
        vector2_type phi2 = phi21 + phi23;

        // Compound terms
        vector_type I = hlsl::promote<vector_type>(0.0);
        vector2_type R123 = R12*R23;
        vector2_type r123 = hlsl::sqrt(R123);
        vector2_type Rs = T121*T121*R23 / (hlsl::promote<vector2_type>(1.0) - R123);

        // Reflectance term for m=0 (DC term amplitude)
        vector2_type C0 = R12 + Rs;
        vector_type S0 = evalSensitivity(0.0, 0.0);
        I += depol(C0) * S0;

        // Reflectance term for m>0 (pairs of diracs)
        vector2_type Cm = Rs - T121;
        NBL_UNROLL for (uint32_t m = 1; m <= 3; m++)
        {
            Cm *= r123;
            vector_type SmS = scalar_type(2.0) * evalSensitivity(scalar_type(m) * OPD, scalar_type(m) * phi2.x);
            vector_type SmP = scalar_type(2.0) * evalSensitivity(scalar_type(m) * OPD, scalar_type(m) * phi2.y);
            I += depolColor(Cm.x*SmS, Cm.y*SmP);
        }

        // Convert back to RGB reflectance
        return hlsl::clamp(hlsl::mul(colorspace::decode::XYZtoscRGB, I), vector_type(0.0), vector_type(1.0));
    }

    scalar_type cosTheta;   // LdotH
    scalar_type Dinc;       // thickness of thin film in micrometers, max 25
    scalar_type ior1;       // thin-film index
    scalar_type ior2;       // complex conductor index, k==0 makes dielectric
    scalar_type iork2;
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
