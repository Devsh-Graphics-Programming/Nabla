// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

// returns unnormalized vector
// TODO: template these?
float computeUnnormalizedMicrofacetNormal(bool _refract, float3 V, float3 L, float orientedEta)
{
    const float etaFactor = (_refract ? orientedEta : 1.0);
    const float3 tmpH = V + L * etaFactor;
    return _refract ? (-tmpH) : tmpH;
}
// returns normalized vector, but NaN when 
float3 computeMicrofacetNormal(bool _refract, float3 V, float3 L, float orientedEta)
{
    const float3 H = computeUnnormalizedMicrofacetNormal(_refract,V,L,orientedEta);
    const float unnormRcpLen = rsqrt(dot(H,H));
    return H * unnormRcpLen;
}

// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
bool isTransmissionPath(float NdotV, float NdotL)
{
    return bool((asuint(NdotV) ^ asuint(NdotL)) & 0x80000000u);
}

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
        retval.direction = math::reflect<T>(direction,N,directionDotN);
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
        matrix<U, 2, 3> TB = math::frisvad<U>(isotropic.N);
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


#define NBL_CONCEPT_NAME Sample
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(I)(A)(B)(U)      // I type Isotropic, A type Aniso, B type Basic (getting clunky)
#define NBL_CONCEPT_PARAM_0 (sample_,T)
#define NBL_CONCEPT_PARAM_1 (iso,I)
#define NBL_CONCEPT_PARAM_2 (aniso,A)
#define NBL_CONCEPT_PARAM_3 (rdirinfo,B)
#define NBL_CONCEPT_PARAM_4 (pV,vector<U, 3>)
#define NBL_CONCEPT_PARAM_5 (frame,matrix<U, 3, 3>)
#define NBL_CONCEPT_PARAM_6 (pT,vector<U, 3>)
#define NBL_CONCEPT_PARAM_7 (pB,vector<U, 3>)
#define NBL_CONCEPT_PARAM_8 (pN,vector<U, 3>)
#define NBL_CONCEPT_PARAM_9 (pVdotL,U)
NBL_CONCEPT_BEGIN(12)
#define sample_ NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define pV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define frame NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define pT NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define pB NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define pN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
#define pVdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_9
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.L), ::nbl::hlsl::is_same_v, B))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.VdotL), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.TdotL), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.BdotL), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.NdotL), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.NdotL2), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createTangentSpace(pV,rdirinfo,frame)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pN)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pT,pB,pN)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<B>(pV,iso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<B>(pV,aniso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sample_.getTangentSpaceL()), ::nbl::hlsl::is_same_v, vector<U,3>))
) && surface_interactions::Anisotropic<A, I, B, U> && surface_interactions::Isotropic<I, B, U> &&
    ray_dir_info::Basic<B, U>;
#undef pVdotL
#undef pN
#undef pB
#undef pT
#undef frame
#undef pV
#undef rdirinfo
#undef aniso
#undef iso
#undef sample_
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo, typename U NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo, U>)
struct SLightSample
{
    using vector_t = vector<U, 3>;
    using matrix_t = matrix<U, 3, 3>;

    static SLightSample<RayDirInfo, U> createTangentSpace(
        NBL_CONST_REF_ARG(vector_t) tangentSpaceV,
        NBL_CONST_REF_ARG(RayDirInfo) tangentSpaceL,
        NBL_CONST_REF_ARG(matrix_t) tangentFrame // WARNING: its the transpose of the old GLSL function return value!
    )
    {
        SLightSample<RayDirInfo, U> retval;
        
        retval.L = RayDirInfo::transform(tangentSpaceL,tangentFrame);
        retval.VdotL = dot(tangentSpaceV,tangentSpaceL);

        retval.TdotL = tangentSpaceL.x;
        retval.BdotL = tangentSpaceL.y;
        retval.NdotL = tangentSpaceL.z;
        retval.NdotL2 = retval.NdotL*retval.NdotL;
        
        return retval;
    }
    static SLightSample<RayDirInfo, U> create(NBL_CONST_REF_ARG(RayDirInfo) L, const U VdotL, NBL_CONST_REF_ARG(vector_t) N)
    {
        SLightSample<RayDirInfo, U> retval;
        
        retval.L = L;
        retval.VdotL = VdotL;

        retval.TdotL = nbl::hlsl::numeric_limits<U>::signaling_NaN;
        retval.BdotL = nbl::hlsl::numeric_limits<U>::signaling_NaN;
        retval.NdotL = dot(N,L);
        retval.NdotL2 = retval.NdotL * retval.NdotL;
        
        return retval;
    }
    static SLightSample<RayDirInfo, U> create(NBL_CONST_REF_ARG(RayDirInfo) L, const U VdotL, NBL_CONST_REF_ARG(vector_t) T, NBL_CONST_REF_ARG(vector_t) B, NBL_CONST_REF_ARG(vector_t) N)
    {
        SLightSample<RayDirInfo, U> retval = create(L,VdotL,N);
        
        retval.TdotL = dot(T,L);
        retval.BdotL = dot(B,L);
        
        return retval;
    }
    // overloads for surface_interactions
    template<class ObserverRayDirInfo>
    static SLightSample<RayDirInfo, U> create(NBL_CONST_REF_ARG(vector_t) L, NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo, U>) interaction)
    {
        const vector_t V = interaction.V.getDirection();
        const float VdotL = dot(V,L);
        return create(L, VdotL, interaction.N);
    }
    template<class ObserverRayDirInfo>
    static SLightSample<RayDirInfo, U> create(NBL_CONST_REF_ARG(vector_t) L, NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo, U>) interaction)
    {
        const vector_t V = interaction.V.getDirection();
        const float VdotL = dot(V,L);
        return create(L,VdotL,interaction.T,interaction.B,interaction.N);
    }
    //
    vector_t getTangentSpaceL()
    {
        return vector_t(TdotL, BdotL, NdotL);
    }

    RayDirInfo L;
    U VdotL;

    U TdotL; 
    U BdotL;
    U NdotL;
    U NdotL2;
};


// everything after here needs testing because godbolt timeout
#define NBL_CONCEPT_NAME IsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)(typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(S)(I)(B)(C)(U)         // S type Sample, I type Isotropic, B/C type Basic (getting clunky)
#define NBL_CONCEPT_PARAM_0 (cache,T)
#define NBL_CONCEPT_PARAM_1 (iso,I)
#define NBL_CONCEPT_PARAM_2 (pNdotV,U)
#define NBL_CONCEPT_PARAM_3 (pNdotL,U)
#define NBL_CONCEPT_PARAM_4 (pVdotL,U)
#define NBL_CONCEPT_PARAM_5 (rcplen,U)
#define NBL_CONCEPT_PARAM_6 (sample_,S)
#define NBL_CONCEPT_PARAM_7 (V,vector<U, 3>)
#define NBL_CONCEPT_PARAM_8 (L,vector<U, 3>)
#define NBL_CONCEPT_PARAM_9 (N,vector<U, 3>)
#define NBL_CONCEPT_PARAM_10 (H,vector<U, 3>)
#define NBL_CONCEPT_PARAM_11 (eta0,U)
#define NBL_CONCEPT_PARAM_12 (eta1,U)
#define NBL_CONCEPT_PARAM_13 (b0,bool)
#define NBL_CONCEPT_PARAM_14 (b1,bool)
NBL_CONCEPT_BEGIN(11)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define pVdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define rcplen NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define sample_ NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define L NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
#define N NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_9
#define H NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_10
#define eta0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_11
#define eta1 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_12
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_13
#define b1 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_14
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.VdotH), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.LdotH), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.NdotH), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.NdotH2), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotL,pVdotL,rcplen)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotL,pVdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<B,C>(iso,sample_)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::compute(cache,b0,V,L,N,pNdotL,pVdotL,eta0,eta1,H)), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<B,C>(cache,iso,sample_,eta0,H)), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<B,C>(cache,iso,sample_,eta0)), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.isValidVNDFMicrofacet(b0,b1,pVdotL,eta0,eta1)), ::nbl::hlsl::is_same_v, bool))
) && surface_interactions::Isotropic<I, B, U> &&
    ray_dir_info::Basic<B, U> && ray_dir_info::Basic<C, U>;
#undef b1
#undef b0
#undef eta1
#undef eta0
#undef H
#undef N
#undef L
#undef V
#undef sample_
#undef rcplen
#undef pVdotL
#undef pNdotL
#undef pNdotV
#undef iso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SIsotropicMicrofacetCache
{
    using vector_t = vector<T, 3>;

    // always valid because its specialized for the reflective case
    static SIsotropicMicrofacetCache<T> createForReflection(const T NdotV, const T NdotL, const T VdotL, out T LplusV_rcpLen)
    {
        LplusV_rcpLen = rsqrt(2.0 + 2.0 * VdotL);

        SIsotropicMicrofacetCache<T> retval;
        
        retval.VdotH = LplusV_rcpLen * VdotL + LplusV_rcpLen;
        retval.LdotH = retval.VdotH;
        retval.NdotH = (NdotL + NdotV) * LplusV_rcpLen;
        retval.NdotH2 = retval.NdotH * retval.NdotH;
        
        return retval;
    }
    static SIsotropicMicrofacetCache<T> createForReflection(const T NdotV, const T NdotL, const T VdotL)
    {
        float dummy;
        return createForReflection(NdotV, NdotL, VdotL, dummy);
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static SIsotropicMicrofacetCache<T> createForReflection(
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo, T>) interaction,
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo, T>) _sample)
    {
        return createForReflection(interaction.NdotV, _sample.NdotL, _sample.VdotL);
    }
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        out SIsotropicMicrofacetCache<T> retval,
        const bool transmitted, NBL_CONST_REF_ARG(vector_t) V, NBL_CONST_REF_ARG(vector_t) L,
        NBL_CONST_REF_ARG(vector_t) N, const T NdotL, const T VdotL,
        const T orientedEta, const T rcpOrientedEta, out vector_t H
    )
    {
        // TODO: can we optimize?
        H = computeMicrofacetNormal(transmitted,V,L,orientedEta);
        retval.NdotH = dot(N, H);
        
        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
        const bool valid = !transmitted || (VdotL <= -min(orientedEta, rcpOrientedEta) && retval.NdotH > nbl::hlsl::numeric_limits<T>::min());
        if (valid)
        {
            // TODO: can we optimize?
            retval.VdotH = dot(V,H);
            retval.LdotH = dot(L,H);
            retval.NdotH2 = retval.NdotH * retval.NdotH;
            return true;
        }
        return false;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        out SIsotropicMicrofacetCache<T> retval,
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo, T>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo, T>) _sample,
        const T eta, out vector_t H
    )
    {
        const T NdotV = interaction.NdotV;
        const T NdotL = _sample.NdotL;
        const bool transmitted = isTransmissionPath(NdotV,NdotL);
        
        float orientedEta, rcpOrientedEta;
        const bool backside = getOrientedEtas<float>(orientedEta,rcpOrientedEta,NdotV,eta);

        const vector_t V = interaction.V.getDirection();
        const vector_t L = _sample.L;
        const float VdotL = dot(V, L);
        return compute(retval,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        out SIsotropicMicrofacetCache<T> retval,
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo, T>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo, T>) _sample,
        const T eta
    )
    {
        vector_t dummy;
        return compute(retval,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,dummy);
    }

    bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const float VdotL, const T eta, const T rcp_eta)
    {
        return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta, rcp_eta)));
    }

    T VdotH;
    T LdotH;
    T NdotH;
    T NdotH2;
};


#define NBL_CONCEPT_NAME AnisotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)(typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(S)(I)(A)(B)(C)(U)         // S type Sample, A type Anisotropic, B/C type Basic (getting clunky)
#define NBL_CONCEPT_PARAM_0 (cache,T)
#define NBL_CONCEPT_PARAM_1 (aniso,I)
#define NBL_CONCEPT_PARAM_2 (pNdotL,U)
#define NBL_CONCEPT_PARAM_3 (pVdotL,U)
#define NBL_CONCEPT_PARAM_4 (sample_,S)
#define NBL_CONCEPT_PARAM_5 (V,vector<U, 3>)
#define NBL_CONCEPT_PARAM_6 (L,vector<U, 3>)
#define NBL_CONCEPT_PARAM_7 (T,vector<U, 3>)
#define NBL_CONCEPT_PARAM_8 (B,vector<U, 3>)
#define NBL_CONCEPT_PARAM_9 (N,vector<U, 3>)
#define NBL_CONCEPT_PARAM_10 (H,vector<U, 3>)
#define NBL_CONCEPT_PARAM_11 (eta0,U)
#define NBL_CONCEPT_PARAM_12 (eta1,U)
#define NBL_CONCEPT_PARAM_13 (b0,bool)
NBL_CONCEPT_BEGIN(11)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define pVdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define sample_ NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define L NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define T NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define B NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
#define N NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_9
#define H NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_10
#define eta0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_11
#define eta1 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_12
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_13
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.TdotH), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.BdotH), ::nbl::hlsl::is_same_v, U))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,H)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,H,b0,eta0,eta1)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(V,L,pVdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<B,C>(aniso,sample_)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::compute(cache,b0,V,L,T,B,N,pNdotL,pVdotL,eta0,eta1,H)), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<B,C>(cache,aniso,sample_)), ::nbl::hlsl::is_same_v, bool))
) && surface_interactions::Anisotropic<A, I, B, U> &&
    ray_dir_info::Basic<B, U> && ray_dir_info::Basic<C, U>;
#undef b0
#undef eta1
#undef eta0
#undef H
#undef N
#undef B
#undef T
#undef L
#undef V
#undef sample_
#undef pVdotL
#undef pNdotL
#undef aniso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SAnisotropicMicrofacetCache : SIsotropicMicrofacetCache<T>
{
    using vector_t = vector<T, 3>;

    // always valid by construction
    static SAnisotropicMicrofacetCache<T> create(NBL_CONST_REF_ARG(vector_t) tangentSpaceV, NBL_CONST_REF_ARG(vector_t) tangentSpaceH)
    {
        SAnisotropicMicrofacetCache<T> retval;
        
        retval.VdotH = dot(tangentSpaceV,tangentSpaceH);
        retval.LdotH = retval.VdotH;
        retval.NdotH = tangentSpaceH.z;
        retval.NdotH2 = retval.NdotH*retval.NdotH;
        retval.TdotH = tangentSpaceH.x;
        retval.BdotH = tangentSpaceH.y;
        
        return retval;
    }
    static SAnisotropicMicrofacetCache<T> create(
        NBL_CONST_REF_ARG(vector_t) tangentSpaceV, 
        NBL_CONST_REF_ARG(vector_t) tangentSpaceH,
        const bool transmitted,
        const T rcpOrientedEta,
        const T rcpOrientedEta2
    )
    {
        SAnisotropicMicrofacetCache<T> retval = create(tangentSpaceV,tangentSpaceH);
        if (transmitted)
        {
            const T VdotH = retval.VdotH;
            LdotH = transmitted ? refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2);
        }
        
        return retval;
    }
    // always valid because its specialized for the reflective case
    static SAnisotropicMicrofacetCache<T> createForReflection(const float3 tangentSpaceV, const float3 tangentSpaceL, const float VdotL)
    {
        SAnisotropicMicrofacetCache<T> retval;
        
        float LplusV_rcpLen;
        retval = createForReflection(tangentSpaceV.z, tangentSpaceL.z, VdotL, LplusV_rcpLen);
        retval.TdotH = (tangentSpaceV.x + tangentSpaceL.x)*LplusV_rcpLen;
        retval.BdotH = (tangentSpaceV.y + tangentSpaceL.y)*LplusV_rcpLen;
        
        return retval;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static SAnisotropicMicrofacetCache<T> createForReflection(
        NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo, T>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo, T>) _sample)
    {
        return createForReflection(interaction.getTangentSpaceV(), _sample.getTangentSpaceL(), _sample.VdotL);
    }
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        out SAnisotropicMicrofacetCache<T> retval,
        const bool transmitted, NBL_CONST_REF_ARG(vector_t) V, NBL_CONST_REF_ARG(vector_t) L,
        NBL_CONST_REF_ARG(vector_t) T, NBL_CONST_REF_ARG(vector_t) B, NBL_CONST_REF_ARG(vector_t) N,
        const T NdotL, const T VdotL,
        const T orientedEta, const T rcpOrientedEta, out vector_t H
    )
    {
        vector_t H;
        const bool valid = SIsotropicMicrofacetCache<T>::compute(retval,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
        if (valid)
        {
            retval.TdotH = dot(T,H);
            retval.BdotH = dot(B,H);
        }
        return valid;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        out SAnisotropicMicrofacetCache<T> retval,
        NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo, T>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo, T>) _sample,
        const T eta
    )
    {
        vector_t H;
        const bool valid = SIsotropicMicrofacetCache<T>::compute(retval,interaction,_sample,eta,H);
        if (valid)
        {
            retval.TdotH = dot(interaction.T,H);
            retval.BdotH = dot(interaction.B,H);
        }
        return valid;
    }

    T TdotH;
    T BdotH;
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
