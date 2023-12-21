// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl> 
#include <nbl/builtin/hlsl/cpp_compat/concepts.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl> 
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl> 
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{


namespace ray_dir_info
{

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(basic, T t)
NBL_CONCEPT_BODY
(
    {T::scalar_t()} -> concepts::scalar;
    {T::vector_t()} -> concepts::vectorial;
    {T::matrix_t()} -> concepts::matricial;

    { t.transform(T::matrix_t()) } -> concepts::same_as<void>;
    { t.getDirection() } -> concepts::same_as<T::vector_t>;
    { t.transmit() } -> concepts::same_as<T>;
    { t.reflect(T::vector_t(), T::scalar_t()) } -> concepts::same_as<T>;
)

// no ray-differentials, nothing
template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
struct Basic
{
    using scalar_t = Scalar;
    using vector_t = vector<Scalar, 3>;
    using matrix_t = matrix<Scalar, 3, 3>;
    
    static Basic create(NBL_CONST_REF_ARG(vector_t) dir)
    {
        Basic retval;
        retval.direction = dir;
        return retval;
    }

    // `tform` must be orthonormal
    void transform(NBL_CONST_REF_ARG(matrix_t) tform)
    {
        direction = tform * direction; // TODO (make sure) is this correct multiplication order in HLSL? Original GLSL code is `tform * direction`
    }

    // `tform` must be orthonormal
    static Basic transform(NBL_CONST_REF_ARG(Basic) r, NBL_CONST_REF_ARG(matrix_t) tform)
    {
        r.transform(tform);
        return r;
    }

    scalar_t getDirection() {return direction;}

    Basic transmit()
    {
        Basic retval;
        retval.direction = -direction;
        return retval;
    }
  
    Basic reflect(NBL_CONST_REF_ARG(vector_t) N, scalar_t directionDotN)
    {
        Basic retval;
        retval.direction = math::reflect(direction,N,directionDotN);
        return retval;
    }

    vector_t direction;
};
// more to come!

}


namespace surface_interactions
{

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(isotropic, T t)
NBL_CONCEPT_BODY
(
    {T::scalar_t()} -> concepts::scalar;
    {T::vector_t()} -> concepts::vectorial;
    {T::matrix_t()} -> concepts::matricial;

    { t.V } -> ray_dir_info::basic;
    { t.N } -> concepts::same_as<T::vector_t>;
    { t.NdotV } -> concepts::same_as<T::scalar_t>;
    { t.NdotV2 } -> concepts::same_as<T::scalar_t>;
)

template <class RayDirInfo>
    NBL_REQUIRES(ray_dir_info::basic<RayDirInfo>)
struct Isotropic
{
    using scalar_t = RayDirInfo::scalar_t;
    using vector_t = RayDirInfo::vector_t;
    using matrix_t = RayDirInfo::matrix_t;
    
    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static Isotropic<RayDirInfo> create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, NBL_CONST_REF_ARG(vector_t) normalizedN)
    {
        Isotropic<RayDirInfo> retval;
        retval.V = normalizedV;
        retval.N = normalizedN;

        retval.NdotV = dot(retval.N,retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;

        return retval;
    }

    RayDirInfo V;
    vector_t N;
    scalar_t NdotV;
    scalar_t NdotV2; // old NdotV_squared
};

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(anisotropic, T t)
NBL_CONCEPT_BODY
(
    // Anisotropic surface interaction conforms to the isotropic concept as well.
    { t } -> isotropic;

    { t.T } -> concepts::same_as<T::vector_t>;
    { t.B } -> concepts::same_as<T::vector_t>;
    { t.TdotV } -> concepts::same_as<T::scalar_t>;
    { t.BdotV } -> concepts::same_as<T::scalar_t>;
    { t.getTangentSpaceV() } -> concepts::same_as<T::vector_t>;
    { t.getTangentFrame() } -> concepts::same_as<T::matrix_t>;
)

template<class RayDirInfo>
    NBL_REQUIRES(ray_dir_info::basic<RayDirInfo>)
struct Anisotropic : Isotropic<RayDirInfo>
{
    using scalar_t = Isotropic<RayDirInfo>::scalar_t;
    using vector_t = Isotropic<RayDirInfo>::vector_t;
    using matrix_t = Isotropic<RayDirInfo>::matrix_t;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static Anisotropic<RayDirInfo> create(
        NBL_CONST_REF_ARG(Isotropic<RayDirInfo>) isotropic,
        NBL_CONST_REF_ARG(vector_t) normalizedT,
        scalar_t normalizedB
    )
    {
        Anisotropic<RayDirInfo> retval;
        (Isotropic<RayDirInfo>) retval = isotropic;
        retval.T = normalizedT;
        retval.B = normalizedB;
    
        const vector_t V = retval.getDirection();
        retval.TdotV = dot(V,retval.T);
        retval.BdotV = dot(V,retval.B);

        return retval;
    }
    
    static Anisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(Isotropic<RayDirInfo>) isotropic, vector_t normalizedT)
    {
        return create(isotropic,normalizedT,cross(isotropic.N,normalizedT));
    }

    static Anisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(Isotropic<RayDirInfo>) isotropic)
    {
        matrix<scalar_t,2,3> TB = math::frisvad(isotropic.N);
        return create(isotropic,TB[0],TB[1]);
    }

    // Not actually tangent space but rather direct product of the tangent space and the
    // space spanned by the normal (which is part of the cotangent/dual space).
    vector_t getTangentSpaceV() {return vector_t(TdotV, BdotV, Isotropic<RayDirInfo>::NdotV);}
    // WARNING: its the transpose of the old GLSL function return value!
    matrix_t getTangentFrame() {return matrix_t(T, B, Isotropic<RayDirInfo>::N);}

    vector_t T;
    vector_t B;
    scalar_t TdotV;
    scalar_t BdotV;
};

}

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(sample, T t)
NBL_CONCEPT_BODY
(
    {T::scalar_t()} -> concepts::scalar;
    {T::vector_t()} -> concepts::vectorial;
    {T::matrix_t()} -> concepts::matricial;

    // Samples should have direction and projections of
    // that direction with the tangent and normal vectors.
    { t.L } -> ray_dir_info::basic;
    { t.TdotL } -> concepts::same_as<T::scalar_t>;
    { t.BdotL } -> concepts::same_as<T::scalar_t>;
    { t.NdotL } -> concepts::same_as<T::scalar_t>;
    { t.NdotL2 } -> concepts::same_as<T::scalar_t>;

    { t.getTangentSpaceL() } -> concepts::same_as<T::vector_t>;
)


template<class RayDirInfo>
    NBL_REQUIRES(ray_dir_info::basic<RayDirInfo>)
struct LightSample
{
    using scalar_t = RayDirInfo::scalar_t;
    using vector_t = RayDirInfo::vector_t;
    using matrix_t = RayDirInfo::matrix_t;
    
    static LightSample<RayDirInfo> createTangentSpace(
        NBL_CONST_REF_ARG(vector_t) tangentSpaceV,
        NBL_CONST_REF_ARG(RayDirInfo) tangentSpaceL,
        NBL_CONST_REF_ARG(matrix_t) tangentFrame // WARNING: its the transpose of the old GLSL function return value!
    )
    {
        LightSample<RayDirInfo> retval;
    
        retval.L = RayDirInfo::transform(tangentSpaceL,tangentFrame);
        retval.VdotL = dot(tangentSpaceV,tangentSpaceL);

        retval.TdotL = tangentSpaceL.x;
        retval.BdotL = tangentSpaceL.y;
        retval.NdotL = tangentSpaceL.z;
        retval.NdotL2 = retval.NdotL*retval.NdotL;
    
        return retval;
    }
    
    static LightSample<RayDirInfo> create(NBL_CONST_REF_ARG(RayDirInfo) L, scalar_t VdotL, NBL_CONST_REF_ARG(vector_t) N)
    {
        LightSample<RayDirInfo> retval;
    
        retval.L = L;
        retval.VdotL = VdotL;

        retval.TdotL = nbl::hlsl::numeric_limits<float>::nan();
        retval.BdotL = nbl::hlsl::numeric_limits<float>::nan();
        retval.NdotL = dot(N,L);
        retval.NdotL2 = retval.NdotL*retval.NdotL;
    
        return retval;
    }
    
    static LightSample<RayDirInfo> create(NBL_CONST_REF_ARG(RayDirInfo) L, scalar_t VdotL, vector_t T, vector_t B, vector_t N)
    {
        LightSample<RayDirInfo> retval = create(L,VdotL,N);
    
        retval.TdotL = dot(T,L);
        retval.BdotL = dot(B,L);
    
        return retval;
    }
    
    // overloads for surface_interactions
    template<class ObserverRayDirInfo>
    static LightSample<RayDirInfo> create(NBL_CONST_REF_ARG(vector_t) L, NBL_CONST_REF_ARG(surface_interactions::Isotropic<ObserverRayDirInfo>) interaction)
    {
        const vector_t V = interaction.V.getDirection();
        const scalar_t VdotL = dot(V,L);
        return create(L,VdotL,interaction.N);
    }

    template<class ObserverRayDirInfo>
    static LightSample<RayDirInfo> create(NBL_CONST_REF_ARG(vector_t) L, NBL_CONST_REF_ARG(surface_interactions::Anisotropic<ObserverRayDirInfo>) interaction)
    {
        const vector_t V = interaction.V.getDirection();
        const scalar_t VdotL = dot(V,L);
        return create(L,VdotL,interaction.T,interaction.B,interaction.N);
    }
    
    vector_t getTangentSpaceL()
    {
        return vector_t(TdotL,BdotL,NdotL);
    }

    RayDirInfo L;
    scalar_t VdotL;

    scalar_t TdotL; 
    scalar_t BdotL;
    scalar_t NdotL;
    scalar_t NdotL2;
};


NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(isotropic_microfacet_cache, T t)
NBL_CONCEPT_BODY
(
    {T::scalar_t()} -> concepts::scalar;
    {T::vector_t()} -> concepts::vectorial;
    {T::matrix_t()} -> concepts::matricial;

    { t.VdotH } -> concepts::same_as<T::scalar_t>;
    { t.LdotH } -> concepts::same_as<T::scalar_t>;
    { t.NdotH } -> concepts::same_as<T::scalar_t>;
    { t.NdotH2 } -> concepts::same_as<T::scalar_t>;
)


template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
struct IsotropicMicrofacetCache
{
    using scalar_t = Scalar;
    using vector_t = vector<Scalar, 3>;
    using matrix_t = matrix<Scalar, 3, 3>;

    // always valid because its specialized for the reflective case
    static IsotropicMicrofacetCache createForReflection(scalar_t NdotV, scalar_t NdotL, scalar_t VdotL, NBL_REF_ARG(scalar_t) LplusV_rcpLen)
    {
        LplusV_rcpLen = rsqrt(2.0+2.0*VdotL);

        IsotropicMicrofacetCache retval;
    
        retval.VdotH = LplusV_rcpLen*VdotL+LplusV_rcpLen;
        retval.LdotH = retval.VdotH;
        retval.NdotH = (NdotL+NdotV)*LplusV_rcpLen;
        retval.NdotH2 = retval.NdotH*retval.NdotH;
    
        return retval;
    }
    
    static IsotropicMicrofacetCache createForReflection(scalar_t NdotV, scalar_t NdotL, scalar_t VdotL)
    {
        scalar_t dummy;
        return createForReflection(NdotV,NdotL,VdotL,dummy);
    }
    
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static IsotropicMicrofacetCache createForReflection(
        const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
        const LightSample<IncomingRayDirInfo> _sample)
    {
        return createForReflection(interaction.NdotV,_sample.NdotL,_sample.VdotL);
    }
    
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        NBL_REF_ARG(IsotropicMicrofacetCache) retval,
        bool transmitted, vector_t V, vector_t L,
        vector_t N, vector_t NdotL, vector_t VdotL,
        scalar_t orientedEta, scalar_t rcpOrientedEta, NBL_REF_ARG(vector_t) H)
    {
        // TODO: can we optimize?
        H = math::computeMicrofacetNormal(transmitted,V,L,orientedEta);
        retval.NdotH = dot(N,H);
    
        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
        const bool valid = !transmitted || (VdotL<=-min(orientedEta,rcpOrientedEta) && retval.NdotH>nbl::hlsl::numeric_limits<float>::min());
        if (valid)
        {
            // TODO: can we optimize?
            retval.VdotH = dot(V,H);
            retval.LdotH = dot(L,H);
            retval.NdotH2 = retval.NdotH*retval.NdotH;
            return true;
        }
        return false;
    }
    
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        NBL_REF_ARG(IsotropicMicrofacetCache) retval,
        NBL_CONST_REF_ARG(surface_interactions::Isotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(LightSample<IncomingRayDirInfo>) _sample,
        const scalar_t eta, NBL_REF_ARG(vector_t) H
    )
    {
        const scalar_t NdotV = interaction.NdotV;
        const scalar_t NdotL = _sample.NdotL;
        const bool transmitted = math::isTransmissionPath(NdotV,NdotL);
    
        float orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas(orientedEta,rcpOrientedEta,NdotV,eta);

        const vector_t V = interaction.V.getDirection();
        const vector_t L = _sample.L;
        const vector_t VdotL = dot(V,L);
        return nbl_glsl_calcIsotropicMicrofacetCache(retval,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
    }

    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        NBL_REF_ARG(IsotropicMicrofacetCache) retval,
        NBL_CONST_REF_ARG(surface_interactions::Isotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(LightSample<IncomingRayDirInfo>) _sample,
        scalar_t eta
    )
    {
        vector_t dummy;
        //float3 V = interaction.V.getDirection();
        //return nbl_glsl_calcIsotropicMicrofacetCache(retval,transmitted,V,L,interaction.N,NdotL,_sample.VdotL,orientedEta,rcpOrientedEta,dummy);
        // TODO try to use other function that doesnt need `dummy`
        //  (computing H that is then lost)
        return compute(retval, interaction, _sample, eta, dummy);
    }

    bool isValidVNDFMicrofacet(bool is_bsdf, bool transmission, scalar_t VdotL, scalar_t eta, scalar_t rcp_eta)
    {
        return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta,rcp_eta)));
    }

    scalar_t VdotH;
    scalar_t LdotH;
    scalar_t NdotH;
    scalar_t NdotH2;
};

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(anisotropic_microfacet_cache, T t)
NBL_CONCEPT_BODY
(
    { t } -> isotropic_microfacet_cache;

    { t.TdotH } -> concepts::same_as<T::scalar_t>;
    { t.BdotH } -> concepts::same_as<T::scalar_t>;
)

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
struct AnisotropicMicrofacetCache : IsotropicMicrofacetCache<Scalar>
{
    using scalar_t = Scalar;
    using vector_t = vector<Scalar, 3>;
    using matrix_t = matrix<Scalar, 3, 3>;

    // always valid by construction
    static AnisotropicMicrofacetCache create(vector_t tangentSpaceV, vector_t tangentSpaceH)
    {
        AnisotropicMicrofacetCache retval;
    
        retval.VdotH = dot(tangentSpaceV,tangentSpaceH);
        retval.LdotH = retval.VdotH;
        retval.NdotH = tangentSpaceH.z;
        retval.NdotH2 = retval.NdotH*retval.NdotH;
        retval.TdotH = tangentSpaceH.x;
        retval.BdotH = tangentSpaceH.y;
    
        return retval;
    }

    static AnisotropicMicrofacetCache create(
        vector_t tangentSpaceV,
        vector_t tangentSpaceH,
        bool transmitted,
        scalar_t rcpOrientedEta,
        scalar_t rcpOrientedEta2
    )
    {
        AnisotropicMicrofacetCache retval = create(tangentSpaceV,tangentSpaceH);
        if (transmitted)
        {
        const scalar_t VdotH = retval.VdotH;
        retval.LdotH = transmitted ? math::refractComputeNdotT(VdotH<0.0, VdotH * VdotH, rcpOrientedEta2) : VdotH;
        }
    
        return retval;
    }

    // always valid because its specialized for the reflective case
    static AnisotropicMicrofacetCache createForReflection(vector_t tangentSpaceV, vector_t tangentSpaceL, scalar_t VdotL)
    {
        AnisotropicMicrofacetCache retval;
    
        scalar_t LplusV_rcpLen;
        (IsotropicMicrofacetCache) retval = IsotropicMicrofacetCache::createForReflection(tangentSpaceV.z,tangentSpaceL.z,VdotL,LplusV_rcpLen);
        retval.TdotH = (tangentSpaceV.x+tangentSpaceL.x)*LplusV_rcpLen;
        retval.BdotH = (tangentSpaceV.y+tangentSpaceL.y)*LplusV_rcpLen;
        
        scalar_t retval;
    }

    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static AnisotropicMicrofacetCache createForReflection(
        NBL_CONST_REF_ARG(surface_interactions::Anisotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(LightSample<IncomingRayDirInfo>) _sample)
    {
        return createForReflection(interaction.getTangentSpaceV(),_sample.getTangentSpaceL(),_sample.VdotL);
    }

    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        NBL_REF_ARG(AnisotropicMicrofacetCache) retval,
        bool transmitted, vector_t V, vector_t L,
        vector_t T, vector_t B, vector_t N,
        scalar_t NdotL, scalar_t VdotL,
        scalar_t orientedEta, scalar_t rcpOrientedEta, NBL_REF_ARG(vector_t) H)
    {
        const bool valid = IsotropicMicrofacetCache::compute(retval, transmitted, V, L, N, NdotL, VdotL, orientedEta, rcpOrientedEta, H);
        if (valid)
        {
        retval.TdotH = dot(T,H);
        retval.BdotH = dot(B,H);
        }
        return valid;
    }

    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        NBL_REF_ARG(AnisotropicMicrofacetCache) retval,
        NBL_CONST_REF_ARG(surface_interactions::Anisotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(LightSample<IncomingRayDirInfo>) _sample,
        scalar_t eta)
    {
        vector_t H;
        const bool valid = IsotropicMicrofacetCache::compute(retval,interaction,_sample,eta,H);
        if (valid)
        {
            retval.TdotH = dot(interaction.T,H);
            retval.BdotH = dot(interaction.B,H);
        }
        return valid;
    }

    scalar_t TdotH;
    scalar_t BdotH;
};

NBL_CONCEPT_TYPE_PARAMS(typename Spec, typename ScalarField)
NBL_CONCEPT_SIGNATURE(generalized_spectral_of, Spec s, ScalarField f)
NBL_CONCEPT_BODY
(
    { f } -> concepts::scalar;
    // Spectrum should be indexable by wavelength
    { s[f] } -> concepts::scalar;
    { f * s } -> concepts::same_as<Spec>;
    { s * f } -> concepts::same_as<Spec>;
)

NBL_CONCEPT_TYPE_PARAMS(typename Spec, typename ScalarField)
NBL_CONCEPT_ASSIGN(spectral_of, generalized_spectral_of<Spec, ScalarField> || concepts::vectorial<Spec> || concepts::scalar<Spec>)

// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename SpectralBins, typename Pdf>
    NBL_REQUIRES(spectral_of<SpectralBins, Pdf> && concepts::floating_point<Pdf>)
struct quotient_and_pdf
{
    static quotient_and_pdf<SpectralBins,Pdf> create(NBL_CONST_REF_ARG(SpectralBins) _quotient, NBL_CONST_REF_ARG(Pdf) _pdf)
    {
        quotient_and_pdf<SpectralBins,Pdf> retval;
        retval.quotient = _quotient;
        retval.pdf = _pdf;
        return retval;
    }

    SpectralBins value()
    {
        return quotient*pdf;
    }
  
    SpectralBins quotient;
    Pdf pdf;
};

typedef quotient_and_pdf<float32_t, float32_t> quotient_and_pdf_scalar;
typedef quotient_and_pdf<vector<float32_t, 3>, float32_t> quotient_and_pdf_rgb;

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(bxdf, T t)
NBL_CONCEPT_BODY
(
    { T::pdf_t() } -> concepts::floating_point;
    { T::spectrum_t() } -> spectral_of<T::pdf_t>;
    { T::q_pdf_t() } -> concepts::same_as<quotient_and_pdf<T::spectrum_t, T::pdf_t>>;

    { T::sample_t() } -> sample;
    // Anisotropic ones will also conform to this concept.
    { T::interaction_t() } -> surface_interactions::isotropic;
)

NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(basic_bxdf, T t)
NBL_CONCEPT_BODY
(
    { t } -> bxdf;

    { t.eval(T::sample_t(), T::interaction_t()) } -> concepts::same_as<T::spectrum_t>;
    { t.generate(T::interaction_t(), T::interaction_t::vector_t()) } -> concepts::same_as<T::sample_t>;
    { t.quotient_and_pdf(T::sample_t(), T::interaction_t()) } -> concepts::same_as<T::q_pdf_t>;
)


NBL_CONCEPT_TYPE_PARAMS(typename T)
NBL_CONCEPT_SIGNATURE(microfacet_bxdf, T t)
NBL_CONCEPT_BODY
(
    // The microfacet_bxdf concept should also conform to bxdf.
    { t } -> bxdf;

    { T::cache_t() } -> isotropic_microfacet_cache;

    { t.eval(T::sample_t(), T::interaction_t(), T::cache_t()) } -> concepts::same_as<T::spectrum_t>;
    { t.generate(T::interaction_t(), T::interaction_t::vector_t(), T::cache_t()) } -> concepts::same_as<T::sample_t>;
    { t.quotient_and_pdf(T::sample_t(), T::interaction_t(), T::cache_t()) } -> concepts::same_as<T::q_pdf_t>;
)


}
}
}

#endif