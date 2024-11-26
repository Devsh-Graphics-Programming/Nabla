// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/math/functions.glsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{


namespace ray_dir_info
{

#define NBL_CONCEPT_NAME Basic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(U)
#define NBL_CONCEPT_PARAM_0 (rdirinfo,T)
#define NBL_CONCEPT_PARAM_1 (N,vector<U, 3>)
#define NBL_CONCEPT_PARAM_2 (dirDotN,U)
NBL_CONCEPT_BEGIN(4)
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define N NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define dirDotN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.direction), ::nbl::hlsl::is_same_v, vector<U, 3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.getDirection()), ::nbl::hlsl::is_same_v, vector<U, 3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transmit()), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflect(N, dirDotN)), ::nbl::hlsl::is_same_v, T))
) && nbl::hlsl::is_scalar_v<U>;
#undef dirDotN
#undef N
#undef rdirinfo
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// no ray-differentials, nothing
template <typename T>
struct SBasic
{
    using vector_t = vector<T, 3>;
    vector_t getDirection() { return direction; }

    SBasic transmit()
    {
        SBasic retval;
        retval.direction = -direction;
        return retval;
    }
    
    SBasic reflect(const float3 N, const float directionDotN)
    {
        SBasic retval;
        retval.direction = nbl::hlsl::reflect(direction,N,directionDotN);   // TODO: template
        return retval;
    }

    vector_t direction;
};
// more to come!

}


namespace surface_interactions
{

#define NBL_CONCEPT_NAME Isotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(B)(U)                        // B is type Basic<T>
#define NBL_CONCEPT_PARAM_0 (iso,T)
#define NBL_CONCEPT_PARAM_1 (normV,B)
#define NBL_CONCEPT_PARAM_2 (normN,vector<U, 3>)
NBL_CONCEPT_BEGIN(5)
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define normV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.V), ::nbl::hlsl::is_same_v, B))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.N), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.NdotV), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.NdotV2), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(normV,normN)), ::nbl::hlsl::is_same_v, T))
) && ray_dir_info::Basic<B, U>;
#undef normN
#undef normV
#undef iso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo, typename T NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo, T>)
struct SIsotropic
{
    using vector_t = vector<T, 3>;
    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static SIsotropic<RayDirInfo, T> create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, NBL_CONST_REF_ARG(vector_t) normalizedN)
    {
        SIsotropic<RayDirInfo, T> retval;
        retval.V = normalizedV;
        retval.N = normalizedN;

        retval.NdotV = dot(retval.N,retval.V.getDirection());
        retval.NdotV2 = retval.NdotV*retval.NdotV;

        return retval;
    }

    RayDirInfo V;
    vector_t N;
    T NdotV;
    T NdotV2; // old NdotV_squared
};

#define NBL_CONCEPT_NAME Anisotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(I)(B)(U)                     // I is type Isotropic<B, T>, B is type Basic<T>
#define NBL_CONCEPT_PARAM_0 (aniso,T)
#define NBL_CONCEPT_PARAM_1 (iso,I)
#define NBL_CONCEPT_PARAM_2 (normT,vector<U, 3>)
#define NBL_CONCEPT_PARAM_3 (normB,U)
NBL_CONCEPT_BEGIN(9)
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normT NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define normB NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.T), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.B), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.TdotV), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.BdotV), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso,normT,normB)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso,normT)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTangentSpaceV()), ::nbl::hlsl::is_same_v, vector<U,3>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTangentFrame()), ::nbl::hlsl::is_same_v, matrix<U,3,3>))
) && Isotropic<I, B, U> && ray_dir_info::Basic<B, U>;
#undef normB
#undef normT
#undef iso
#undef aniso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo, typename U NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo, U>)
struct SAnisotropic : SIsotropic<RayDirInfo, U>
{
    using vector_t = vector<U, 3>;
    using matrix_t = matrix<U, 3, 3>;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static SAnisotropic<RayDirInfo, U> create(
        NBL_CONST_REF_ARG(SIsotropic<RayDirInfo, U>) isotropic,
        NBL_CONST_REF_ARG(vector_t) normalizedT,
        const U normalizedB
    )
    {
        SAnisotropic<RayDirInfo, U> retval;
        //(SIsotropic<RayDirInfo, U>) retval = isotropic;
        retval.T = normalizedT;
        retval.B = normalizedB;
        
        const vector_t V = retval.getDirection();
        retval.TdotV = dot(V, retval.T);
        retval.BdotV = dot(V, retval.B);

        return retval;
    }
    static SAnisotropic<RayDirInfo, U> create(NBL_CONST_REF_ARG(SIsotropic<RayDirInfo, U>) isotropic, NBL_CONST_REF_ARG(vector_t) normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.N, normalizedT));
    }
    static SAnisotropic<RayDirInfo, U> create(NBL_CONST_REF_ARG(SIsotropic<RayDirInfo, U>) isotropic)
    {
        matrix<U, 2, 3> TB = nbl::hlsl::frisvad(isotropic.N);   // TODO: template
        return create(isotropic, TB[0], TB[1]);
    }

    vector_t getTangentSpaceV() { return vector_t(TdotV, BdotV, SIsotropic<RayDirInfo, U>::NdotV); }
    // WARNING: its the transpose of the old GLSL function return value!
    matrix_t getTangentFrame() { return matrix_t(T, B, SIsotropic<RayDirInfo, U>::N); }

    vector_t T;
    vector_t B;
    vector_t TdotV;
    vector_t BdotV;
};

}


template<class RayDirInfo>
struct LightSample
{
    static LightSample<RayDirInfo> createTangentSpace(
        const float3 tangentSpaceV,
        const RayDirInfo tangentSpaceL,
        const float3x3 tangentFrame // WARNING: its the transpose of the old GLSL function return value!
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
    static LightSample<RayDirInfo> create(const RayDirInfo L, const float VdotL, const float3 N)
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
    static LightSample<RayDirInfo> create(const RayDirInfo L, const float VdotL, const float3 T, const float3 B, const float3 N)
    {
        LightSample<RayDirInfo> retval = create(L,VdotL,N);
        
        retval.TdotL = dot(T,L);
        retval.BdotL = dot(B,L);
        
        return retval;
    }
    // overloads for surface_interactions
    template<class ObserverRayDirInfo>
    static LightSample<RayDirInfo> create(const float3 L, const surface_interactions::Isotropic<ObserverRayDirInfo> interaction)
    {
        const float3 V = interaction.V.getDirection();
        const float VdotL = dot(V,L);
        return create(L,VdotL,interaction.N);
    }
    template<class ObserverRayDirInfo>
    static LightSample<RayDirInfo> create(const float3 L, const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction)
    {
        const float3 V = interaction.V.getDirection();
        const float VdotL = dot(V,L);
        return create(L,VdotL,interaction.T,interaction.B,interaction.N);
    }
    //
    float3 getTangentSpaceL()
    {
        return float3(TdotL,BdotL,NdotL);
    }

    RayDirInfo L;
    float VdotL;

    float TdotL; 
    float BdotL;
    float NdotL;
    float NdotL2;
};


//
struct IsotropicMicrofacetCache
{
    // always valid because its specialized for the reflective case
    static IsotropicMicrofacetCache createForReflection(const float NdotV, const float NdotL, const float VdotL, out float LplusV_rcpLen)
    {
        LplusV_rcpLen = inversesqrt(2.0+2.0*VdotL);

        IsotropicMicrofacetCache retval;
        
        retval.VdotH = LplusV_rcpLen*VdotL+LplusV_rcpLen;
        retval.LdotH = retval.VdotH;
        retval.NdotH = (NdotL+NdotV)*LplusV_rcpLen;
        retval.NdotH2 = retval.NdotH*retval.NdotH;
        
        return retval;
    }
    static IsotropicMicrofacetCache createForReflection(const float NdotV, const float NdotL, const float VdotL)
    {
        float dummy;
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
        out IsotropicMicrofacetCache retval,
        const bool transmitted, const float3 V, const float3 L,
        const float3 N, const float NdotL, const float VdotL,
        const float orientedEta, const float rcpOrientedEta, out float3 H
    )
    {
        // TODO: can we optimize?
        H = computeMicrofacetNormal(transmitted,V,L,orientedEta);
        retval.NdotH = dot(N,H);
        
        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
        const bool valid = !transmitted || (VdotL<=-min(orientedEta,rcpOrientedEta) && _cache.NdotH>nbl::hlsl::numeric_limits::min());
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
        out IsotropicMicrofacetCache retval,
        const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
        const LightSample<IncomingRayDirInfo> _sample,
        const float eta, out float3 H
    )
    {
        const float NdotV = interaction.NdotV;
        const float NdotL = _sample.NdotL;
        const bool transmitted = nbl_glsl_isTransmissionPath(NdotV,NdotL);
        
        float orientedEta, rcpOrientedEta;
        const bool backside = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,NdotV,eta);

        const vec3 V = interaction.V.getDirection();
        const vec3 L = _sample.L;
        const float VdotL = dot(V,L);
        return nbl_glsl_calcIsotropicMicrofacetCache(_cache,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        out IsotropicMicrofacetCache retval,
        const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
        const LightSample<IncomingRayDirInfo> _sample,
        const float eta
    )
    {
        float3 dummy;
        return nbl_glsl_calcIsotropicMicrofacetCache(_cache,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,dummy);
    }

    bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const float VdotL, const float eta, const float rcp_eta)
    {
        return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta,rcp_eta)));
    }

    float VdotH;
    float LdotH;
    float NdotH;
    float NdotH2;
};

struct AnisotropicMicrofacetCache : IsotropicMicrofacetCache
{
    // always valid by construction
    static AnisotropicMicrofacetCache create(const float3 tangentSpaceV, const float3 tangentSpaceH)
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
        const float3 tangentSpaceV, 
        const float3 tangentSpaceH,
        const bool transmitted,
        const float rcpOrientedEta,
        const float rcpOrientedEta2
    )
    {
        AnisotropicMicrofacetCache retval = create(tangentSpaceV,tangentSpaceH);
        if (transmitted)
        {
            const float VdotH = retval.VdotH;
            LdotH = transmitted ? refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2);
        }
        
        return retval;
    }
    // always valid because its specialized for the reflective case
    static AnisotropicMicrofacetCache createForReflection(const float3 tangentSpaceV, const float3 tangentSpaceL, const float VdotL)
    {
        AnisotropicMicrofacetCache retval;
        
        float LplusV_rcpLen;
        retval = createForReflection(tangentSpaceV.z,tangentSpaceL.z,VdotL,LplusV_rcpLen);
        retval.TdotH = (tangentSpaceV.x+tangentSpaceL.x)*LplusV_rcpLen;
        retval.BdotH = (tangentSpaceV.y+tangentSpaceL.y)*LplusV_rcpLen;
        
        return retval;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static AnisotropicMicrofacetCache createForReflection(
        const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction, 
        const LightSample<IncomingRayDirInfo> _sample)
    {
        return createForReflection(interaction.getTangentSpaceV(),_sample.getTangentSpaceL(),_sample.VdotL);
    }
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        out AnisotropicMicrofacetCache retval,
        const bool transmitted, const float3 V, const float3 L,
        const float3 T, const float3 B, const float3 N,
        const float NdotL, const float VdotL,
        const float orientedEta, const float rcpOrientedEta, out float3 H
    )
    {
        float3 H;
        const bool valid = IsotropicMicrofacetCache::compute(retval,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
        if (valid)
        {
            retval.TdotH = dot(T,H);
            retval.BdotH = dot(B,H);
        }
        return valid;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        out AnisotropicMicrofacetCache retval,
        const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction, 
        const LightSample<IncomingRayDirInfo> _sample,
        const float eta
    )
    {
        float3 H;
        const bool valid = IsotropicMicrofacetCache::compute(retval,interaction,_sample,eta,H);
        if (valid)
        {
            retval.TdotH = dot(interaction.T,H);
            retval.BdotH = dot(interaction.B,H);
        }
        return valid;
    }

    float TdotH;
    float BdotH;
};


// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename SpectralBins>
struct quotient_and_pdf
{
    quotient_and_pdf<SpectralBins> create(const SpectralBins _quotient, const float _pdf)
    {
        quotient_and_pdf<SpectralBins> retval;
        retval.quotient = _quotient;
        retval.pdf = _pdf;
        return retval;
    }

    SpectralBins value()
    {
        return quotient*pdf;
    }
    
    SpectralBins quotient;
    float pdf;
};


}
}
}

#endif
