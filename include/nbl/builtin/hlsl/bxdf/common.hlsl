// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{

// TODO: move into ieee754 namespace hlsl
namespace ieee754
{
    template<typename T NBL_FUNC_REQUIRES(is_floating_point_v<T>)
    T condNegate(T a, bool flip)
    {
        return flip ? (-a) : a;
    }
}

namespace bxdf
{

// returns unnormalized vector
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T computeUnnormalizedMicrofacetNormal(bool _refract, vector<T,3> V, vector<T,3> L, T orientedEta)
{
    const T etaFactor = (_refract ? orientedEta : 1.0);
    const vector<T,3> tmpH = V + L * etaFactor;
    return ieee754::condNegate<T>(tmpH, _refract);
}

// returns normalized vector, but NaN when result is length 0
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T computeMicrofacetNormal(bool _refract, vector<T,3> V, vector<T,3> L, T orientedEta)
{
    const vector<T,3> H = computeUnnormalizedMicrofacetNormal<T>(_refract,V,L,orientedEta);
    const T unnormRcpLen = rsqrt<T>(nbl::hlsl::dot<T>(H,H));
    return H * unnormRcpLen;
}

// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
bool isTransmissionPath(float NdotV, float NdotL)
{
#ifdef __HLSL_VERSION
    return bool((asuint(NdotV) ^ asuint(NdotL)) & 0x80000000u);
#else
    return bool((bit_cast<uint32_t>(NdotV) ^ bit_cast<uint32_t>(NdotL)) & 0x80000000u);
#endif
}

namespace ray_dir_info
{

#define NBL_CONCEPT_NAME Basic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (rdirinfo, T)
#define NBL_CONCEPT_PARAM_1 (N, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_2 (dirDotN, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (m, typename T::matrix3x3_type)
NBL_CONCEPT_BEGIN(4)
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define N NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define dirDotN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define m NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.direction), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.getDirection()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transmit()), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflect(N, dirDotN)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.refract(N, dirDotN)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::transform(m, rdirinfo)), ::nbl::hlsl::is_same_v, T))
) && is_scalar_v<typename T::scalar_type> && is_vector_v<typename T::vector3_type>;
#undef m
#undef dirDotN
#undef N
#undef rdirinfo
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T>
struct SBasic
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using matrix3x3_type = matrix<T, 3, 3>;

    vector3_type getDirection() NBL_CONST_MEMBER_FUNC { return direction; }

    SBasic<T> transmit()
    {
        SBasic<T> retval;
        retval.direction = -direction;
        return retval;
    }
    
    SBasic<T> reflect(NBL_CONST_REF_ARG(vector3_type) N, scalar_type directionDotN)
    {
        SBasic<T> retval;
        retval.direction = math::reflect<T>(direction,N,directionDotN);
        return retval;
    }

    SBasic<T> refract(NBL_CONST_REF_ARG(vector3_type) N, scalar_type eta)
    {
        SBasic<T> retval;
        retval.direction = math::refract<T>(direction,N,eta);
        return retval;
    }

    // WARNING: matrix must be orthonormal
    static SBasic<T> transform(NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_CONST_REF_ARG(SBasic<T>) r)
    {
#ifndef __HLSL__VERSION
        matrix3x3_type m_T = nbl::hlsl::transpose<matrix3x3_type>(m);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[1])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[2])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[1], m_T[2])) < 1e-5);
#endif

        SBasic<T> retval;
        retval.direction = nbl::hlsl::mul<matrix3x3_type,vector3_type>(m, r.direction);
        return retval;
    }

    vector3_type direction;
};
// more to come!

}


namespace surface_interactions
{

#define NBL_CONCEPT_NAME Isotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (iso, T)
#define NBL_CONCEPT_PARAM_1 (normV, typename T::ray_dir_info_type)
#define NBL_CONCEPT_PARAM_2 (normN, typename T::vector3_type)
NBL_CONCEPT_BEGIN(3)
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define normV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.V), ::nbl::hlsl::is_same_v, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.N), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.NdotV), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.NdotV2), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(normV,normN)), ::nbl::hlsl::is_same_v, T))
) && ray_dir_info::Basic<typename T::ray_dir_info_type>;
#undef normN
#undef normV
#undef iso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo>)
struct SIsotropic
{
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static SIsotropic<RayDirInfo> create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, NBL_CONST_REF_ARG(vector3_type) normalizedN)
    {
        SIsotropic<RayDirInfo> retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = nbl::hlsl::dot<vector3_type>(retval.N, retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;

        return retval;
    }

    RayDirInfo V;
    vector3_type N;
    scalar_type NdotV;
    scalar_type NdotV2;
};

#define NBL_CONCEPT_NAME Anisotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (aniso, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_type)
#define NBL_CONCEPT_PARAM_2 (normT, typename T::vector3_type)
NBL_CONCEPT_BEGIN(3)
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normT NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.T), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.B), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.TdotV), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.BdotV), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso,normT,normT)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTangentSpaceV()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getToTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getFromTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
) && Isotropic<typename T::isotropic_type> && ray_dir_info::Basic<typename T::ray_dir_info_type>;
#undef normT
#undef iso
#undef aniso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo>)
struct SAnisotropic : SIsotropic<RayDirInfo>
{
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;
    using isotropic_type = SIsotropic<RayDirInfo>;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static SAnisotropic<RayDirInfo> create(
        NBL_CONST_REF_ARG(isotropic_type) isotropic,
        NBL_CONST_REF_ARG(vector3_type) normalizedT,
        NBL_CONST_REF_ARG(vector3_type) normalizedB
    )
    {
        SAnisotropic<RayDirInfo> retval;
        //(SIsotropic<RayDirInfo>) retval = isotropic;
        retval.V = isotropic.V;
        retval.N = isotropic.N;
        retval.NdotV = isotropic.NdotV;
        retval.NdotV2 = isotropic.NdotV2;
        
        retval.T = normalizedT;
        retval.B = normalizedB;
        
        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.V.getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.V.getDirection(), retval.B);

        return retval;
    }
    static SAnisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(isotropic_type) isotropic, NBL_CONST_REF_ARG(vector3_type) normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.N, normalizedT));
    }
    static SAnisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(isotropic_type) isotropic)
    {
        matrix<scalar_type, 2, 3> TB = math::frisvad<scalar_type>(isotropic.N);
        return create(isotropic, TB[0], TB[1]);
    }

    vector3_type getTangentSpaceV() { return vector3_type(TdotV, BdotV, isotropic_type::NdotV); }
    matrix3x3_type getToTangentSpace() { return matrix3x3_type(T, B, isotropic_type::N); }
    matrix3x3_type getFromTangentSpace() { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic_type::N)); }

    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};

}


#define NBL_CONCEPT_NAME Sample
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sample, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_type)
#define NBL_CONCEPT_PARAM_2 (aniso, typename T::anisotropic_type)
#define NBL_CONCEPT_PARAM_3 (rdirinfo, typename T::ray_dir_info_type)
#define NBL_CONCEPT_PARAM_4 (pV, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (frame, typename T::matrix3x3_type)
#define NBL_CONCEPT_PARAM_6 (pVdotL, typename T::scalar_type)
NBL_CONCEPT_BEGIN(7)
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define pV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define frame NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define pVdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.L), ::nbl::hlsl::is_same_v, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.VdotL), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.TdotL), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.BdotL), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.NdotL), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.NdotL2), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createFromTangentSpace(pV,rdirinfo,frame)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pV,pV,pV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<typename T::ray_dir_info_type>(pV,iso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<typename T::ray_dir_info_type>(pV,aniso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getTangentSpaceL()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
) && surface_interactions::Anisotropic<typename T::anisotropic_type> && surface_interactions::Isotropic<typename T::isotropic_type> &&
    ray_dir_info::Basic<typename T::ray_dir_info_type>;
#undef pVdotL
#undef frame
#undef pV
#undef rdirinfo
#undef aniso
#undef iso
#undef _sample
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo>)
struct SLightSample
{
    using this_t = SLightSample<RayDirInfo>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    using isotropic_type = surface_interactions::SIsotropic<RayDirInfo>;
    using anisotropic_type = surface_interactions::SAnisotropic<RayDirInfo>;

    static this_t createFromTangentSpace(
        NBL_CONST_REF_ARG(vector3_type) tangentSpaceV,
        NBL_CONST_REF_ARG(RayDirInfo) tangentSpaceL,
        NBL_CONST_REF_ARG(matrix3x3_type) tangentFrame
    )
    {
        this_t retval;

        const vector3_type tsL = tangentSpaceL.getDirection();
        retval.L = ray_dir_info_type::transform(tangentFrame, tangentSpaceL);
        retval.VdotL = nbl::hlsl::dot<vector3_type>(tangentSpaceV, tsL);

        retval.TdotL = tsL.x;
        retval.BdotL = tsL.y;
        retval.NdotL = tsL.z;
        retval.NdotL2 = retval.NdotL*retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) L, const scalar_type VdotL, NBL_CONST_REF_ARG(vector3_type) N)
    {
        this_t retval;

        retval.L = L;
        retval.VdotL = VdotL;

        retval.TdotL = nbl::hlsl::numeric_limits<scalar_type>::signaling_NaN;
        retval.BdotL = nbl::hlsl::numeric_limits<scalar_type>::signaling_NaN;
        retval.NdotL = nbl::hlsl::dot<vector3_type>(N,L.direction);
        retval.NdotL2 = retval.NdotL * retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) L, const scalar_type VdotL, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N)
    {
        this_t retval = create(L,VdotL,N);
        
        retval.TdotL = nbl::hlsl::dot<vector3_type>(T,L.direction);
        retval.BdotL = nbl::hlsl::dot<vector3_type>(B,L.direction);
        
        return retval;
    }
    // overloads for surface_interactions
    template<class ObserverRayDirInfo>
    static this_t create(NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo>) interaction)
    {
        const vector3_type V = interaction.V.getDirection();
        const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V,L);
        return create(L, VdotL, interaction.N);
    }
    template<class ObserverRayDirInfo>
    static this_t create(NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo>) interaction)
    {
        const vector3_type V = interaction.V.getDirection();
        const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V,L);
        return create(L,VdotL,interaction.T,interaction.B,interaction.N);
    }
    //
    vector3_type getTangentSpaceL()
    {
        return vector3_type(TdotL, BdotL, NdotL);
    }

    RayDirInfo L;
    scalar_type VdotL;

    scalar_type TdotL; 
    scalar_type BdotL;
    scalar_type NdotL;
    scalar_type NdotL2;
};


// TODO: figure out the commented constraints, templated RayDirInfo not really working for some reason
#define NBL_CONCEPT_NAME IsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_type)
#define NBL_CONCEPT_PARAM_2 (pNdotV, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (b0, bool)
NBL_CONCEPT_BEGIN(6)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.VdotH), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.LdotH), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.NdotH), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.NdotH2), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, T))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<typename T::ray_dir_info_type,typename T::ray_dir_info_type>(iso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::compute(cache,b0,V,V,V,pNdotV,pNdotV,pNdotV,pNdotV,V)), ::nbl::hlsl::is_same_v, bool))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<typename T::ray_dir_info_type,typename T::ray_dir_info_type>(cache,iso,_sample,pNdotV,V)), ::nbl::hlsl::is_same_v, bool))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<typename T::ray_dir_info_type,typename T::ray_dir_info_type>(cache,iso,_sample,pNdotV)), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.isValidVNDFMicrofacet(b0,b0,pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, bool))
) && surface_interactions::Isotropic<typename T::isotropic_type>;
#undef b0
#undef V
#undef _sample
#undef pNdotV
#undef iso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SIsotropicMicrofacetCache
{
    using this_t = SIsotropicMicrofacetCache<T>;
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    using ray_dir_info_type = ray_dir_info::SBasic<scalar_type>;
    using isotropic_type = surface_interactions::SIsotropic<ray_dir_info_type>;
    using sample_type = SLightSample<ray_dir_info_type>;

    // always valid because its specialized for the reflective case
    static this_t createForReflection(const scalar_type NdotV, const scalar_type NdotL, const scalar_type VdotL, NBL_REF_ARG(scalar_type) LplusV_rcpLen)
    {
        LplusV_rcpLen = rsqrt<scalar_type>(2.0 + 2.0 * VdotL);

        this_t retval;
        
        retval.VdotH = LplusV_rcpLen * VdotL + LplusV_rcpLen;
        retval.LdotH = retval.VdotH;
        retval.NdotH = (NdotL + NdotV) * LplusV_rcpLen;
        retval.NdotH2 = retval.NdotH * retval.NdotH;
        
        return retval;
    }
    static this_t createForReflection(const scalar_type NdotV, const scalar_type NdotL, const scalar_type VdotL)
    {
        float dummy;
        return createForReflection(NdotV, NdotL, VdotL, dummy);
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo NBL_FUNC_REQUIRES(ray_dir_info::Basic<ObserverRayDirInfo> && ray_dir_info::Basic<IncomingRayDirInfo>)
    static this_t createForReflection(
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo>) interaction,
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo>) _sample)
    {
        return createForReflection(interaction.NdotV, _sample.NdotL, _sample.VdotL);
    }
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        NBL_REF_ARG(this_t) retval,
        const bool transmitted, NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) L,
        NBL_CONST_REF_ARG(vector3_type) N, const scalar_type NdotL, const scalar_type VdotL,
        const scalar_type orientedEta, const scalar_type rcpOrientedEta, NBL_REF_ARG(vector3_type) H
    )
    {
        // TODO: can we optimize?
        H = computeMicrofacetNormal<scalar_type>(transmitted,V,L,orientedEta);
        retval.NdotH = nbl::hlsl::dot<vector3_type>(N, H);
        
        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
        const bool valid = !transmitted || (VdotL <= -min(orientedEta, rcpOrientedEta) && retval.NdotH > nbl::hlsl::numeric_limits<scalar_type>::min());
        if (valid)
        {
            // TODO: can we optimize?
            retval.VdotH = nbl::hlsl::dot<vector3_type>(V,H);
            retval.LdotH = nbl::hlsl::dot<vector3_type>(L,H);
            retval.NdotH2 = retval.NdotH * retval.NdotH;
            return true;
        }
        return false;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo NBL_FUNC_REQUIRES(ray_dir_info::Basic<ObserverRayDirInfo> && ray_dir_info::Basic<IncomingRayDirInfo>)
    static bool compute(
        NBL_REF_ARG(this_t) retval,
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo>) _sample,
        const scalar_type eta, NBL_REF_ARG(vector3_type) H
    )
    {
        const scalar_type NdotV = interaction.NdotV;
        const scalar_type NdotL = _sample.NdotL;
        const bool transmitted = isTransmissionPath(NdotV,NdotL);
        
        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta,rcpOrientedEta,NdotV,eta);

        const vector3_type V = interaction.V.getDirection();
        const vector3_type L = _sample.L;
        const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V, L);
        return compute(retval,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo NBL_FUNC_REQUIRES(ray_dir_info::Basic<ObserverRayDirInfo> && ray_dir_info::Basic<IncomingRayDirInfo>)
    static bool compute(
        NBL_REF_ARG(this_t) retval,
        NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo>) _sample,
        const scalar_type eta
    )
    {
        vector3_type dummy;
        return compute<ObserverRayDirInfo, IncomingRayDirInfo>(retval,interaction,_sample,eta,dummy);
    }

    bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const scalar_type VdotL, const scalar_type eta, const scalar_type rcp_eta)
    {
        return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta, rcp_eta)));
    }

    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type NdotH;
    scalar_type NdotH2;
};


#define NBL_CONCEPT_NAME AnisotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (aniso, typename T::anisotropic_type)
#define NBL_CONCEPT_PARAM_2 (pNdotL, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (b0, bool)
NBL_CONCEPT_BEGIN(6)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.TdotH), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.BdotH), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,b0,pNdotL,pNdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(V,V,pNdotL)), ::nbl::hlsl::is_same_v, T))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<typename T::ray_dir_info_type,typename T::ray_dir_info_type>(aniso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::compute(cache,b0,V,V,V,V,V,pNdotL,pNdotL,pNdotL,pNdotL,V)), ::nbl::hlsl::is_same_v, bool))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template compute<typename T::ray_dir_info_type,typename T::ray_dir_info_type>(cache,aniso,_sample)), ::nbl::hlsl::is_same_v, bool))
) && surface_interactions::Anisotropic<typename T::anisotropic_type>;
#undef b0
#undef V
#undef _sample
#undef pNdotL
#undef aniso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename U NBL_PRIMARY_REQUIRES(is_scalar_v<U>)
struct SAnisotropicMicrofacetCache : SIsotropicMicrofacetCache<U>
{
    using this_t = SAnisotropicMicrofacetCache<U>;
    using scalar_type = U;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    using ray_dir_info_type = ray_dir_info::SBasic<scalar_type>;
    using anisotropic_type = surface_interactions::SAnisotropic<ray_dir_info_type>;
    using sample_type = SLightSample<ray_dir_info_type>;

    // always valid by construction
    static this_t create(NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, NBL_CONST_REF_ARG(vector3_type) tangentSpaceH)
    {
        this_t retval;
        
        retval.VdotH = nbl::hlsl::dot<vector3_type>(tangentSpaceV,tangentSpaceH);
        retval.LdotH = retval.VdotH;
        retval.NdotH = tangentSpaceH.z;
        retval.NdotH2 = retval.NdotH*retval.NdotH;
        retval.TdotH = tangentSpaceH.x;
        retval.BdotH = tangentSpaceH.y;
        
        return retval;
    }
    static this_t create(
        NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, 
        NBL_CONST_REF_ARG(vector3_type) tangentSpaceH,
        const bool transmitted,
        const scalar_type rcpOrientedEta,
        const scalar_type rcpOrientedEta2
    )
    {
        this_t retval = create(tangentSpaceV,tangentSpaceH);
        if (transmitted)
        {
            const scalar_type VdotH = retval.VdotH;
            retval.LdotH = transmitted ? refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2) : VdotH;
        }
        
        return retval;
    }
    // always valid because its specialized for the reflective case
    static this_t createForReflection(NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, NBL_CONST_REF_ARG(vector3_type) tangentSpaceL, const scalar_type VdotL)
    {
        this_t retval;
        
        scalar_type LplusV_rcpLen;
        retval = (this_t)SIsotropicMicrofacetCache::createForReflection(tangentSpaceV.z, tangentSpaceL.z, VdotL, LplusV_rcpLen);
        retval.TdotH = (tangentSpaceV.x + tangentSpaceL.x) * LplusV_rcpLen;
        retval.BdotH = (tangentSpaceV.y + tangentSpaceL.y) * LplusV_rcpLen;
        
        return retval;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static this_t createForReflection(
        NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo>) _sample)
    {
        return createForReflection(interaction.getTangentSpaceV(), _sample.getTangentSpaceL(), _sample.VdotL);
    }
    // transmissive cases need to be checked if the path is valid before usage
    static bool compute(
        NBL_REF_ARG(this_t) retval,
        const bool transmitted, NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) L,
        NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N,
        const scalar_type NdotL, const scalar_type VdotL,
        const scalar_type orientedEta, const scalar_type rcpOrientedEta, NBL_REF_ARG(vector3_type) H
    )
    {
        const bool valid = this_t::compute(retval,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(B,H);
        }
        return valid;
    }
    template<class ObserverRayDirInfo, class IncomingRayDirInfo>
    static bool compute(
        NBL_REF_ARG(this_t) retval,
        NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo>) interaction, 
        NBL_CONST_REF_ARG(SLightSample<IncomingRayDirInfo>) _sample,
        const scalar_type eta
    )
    {
        vector3_type H;
        const bool valid = this_t::compute(retval,interaction,_sample,eta,H);
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(interaction.T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(interaction.B,H);
        }
        return valid;
    }

    scalar_type TdotH;
    scalar_type BdotH;
};


#define NBL_CONCEPT_NAME generalized_spectral_of
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(F)
#define NBL_CONCEPT_PARAM_0 (spec, T)
#define NBL_CONCEPT_PARAM_1 (field, F)
NBL_CONCEPT_BEGIN(2)
#define spec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define field NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((spec[field]), ::nbl::hlsl::is_scalar_v))  // correctness?
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((spec * field), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((field * spec), ::nbl::hlsl::is_same_v, T))
) && is_scalar_v<F>;
#undef field
#undef spec
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T, typename F>
NBL_BOOL_CONCEPT spectral_of = generalized_spectral_of<T,F> || is_vector_v<T> || is_scalar_v<T>;

// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename SpectralBins, typename Pdf NBL_PRIMARY_REQUIRES(spectral_of<SpectralBins,Pdf> && is_floating_point_v<Pdf>)
struct quotient_and_pdf
{
    using this_t = quotient_and_pdf<SpectralBins, Pdf>;
    static this_t create(NBL_CONST_REF_ARG(SpectralBins) _quotient, NBL_CONST_REF_ARG(Pdf) _pdf)
    {
        this_t retval;
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


#define NBL_CONCEPT_NAME BxDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (spec, typename T::spectral_type)
#define NBL_CONCEPT_PARAM_2 (pdf, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_4 (iso, typename T::isotropic_type)
#define NBL_CONCEPT_PARAM_5 (aniso, typename T::anisotropic_type)
#define NBL_CONCEPT_PARAM_6 (param, typename T::params_t)
NBL_CONCEPT_BEGIN(7)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define spec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define param NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::params_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,aniso.N)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template pdf<LS,I>(_sample,iso)), ::nbl::hlsl::is_scalar_v))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
) && Sample<typename T::sample_type> && spectral_of<typename T::spectral_type,typename T::scalar_type> &&
    surface_interactions::Isotropic<typename T::isotropic_type> && surface_interactions::Anisotropic<typename T::anisotropic_type>;
#undef param
#undef aniso
#undef iso
#undef _sample
#undef pdf
#undef spec
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME MicrofacetBxDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (spec, typename T::spectral_type)
#define NBL_CONCEPT_PARAM_2 (pdf, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_4 (iso, typename T::isotropic_type)
#define NBL_CONCEPT_PARAM_5 (aniso, typename T::anisotropic_type)
#define NBL_CONCEPT_PARAM_6 (isocache, typename T::isocache_type)
#define NBL_CONCEPT_PARAM_7 (anisocache, typename T::anisocache_type)
#define NBL_CONCEPT_PARAM_8 (param, typename T::params_t)
NBL_CONCEPT_BEGIN(9)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define spec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define isocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define anisocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define param NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param)), ::nbl::hlsl::is_same_v, T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,aniso.N,anisocache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.template pdf<LS,I>(_sample,iso)), ::nbl::hlsl::is_scalar_v))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
) && Sample<typename T::sample_type> && spectral_of<typename T::spectral_type,typename T::scalar_type> &&
    IsotropicMicrofacetCache<typename T::isocache_type> && AnisotropicMicrofacetCache<typename T::anisocache_type>;
#undef param
#undef anisocache
#undef isocache
#undef aniso
#undef iso
#undef _sample
#undef pdf
#undef spec
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

enum BxDFClampMode : uint16_t
{
    BCM_NONE = 0,
    BCM_MAX,
    BCM_ABS
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFParams
{
    using this_t = SBxDFParams<Scalar>;

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    static this_t create(LightSample _sample, Iso interaction, BxDFClampMode clamp = BCM_NONE)
    {
        this_t retval;
        retval.NdotV = clamp == BCM_ABS ? abs<Scalar>(interaction.NdotV) : 
                        clamp == BCM_MAX ? max<Scalar>(interaction.NdotV, 0.0) :
                                        interaction.NdotV;
        retval.uNdotV = interaction.NdotV;
        retval.NdotV2 = interaction.NdotV2;
        retval.NdotL = clamp == BCM_ABS ? abs<Scalar>(_sample.NdotL) :
                        clamp == BCM_MAX ? max<Scalar>(_sample.NdotL, 0.0) :
                                        _sample.NdotL;
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotL = _sample.VdotL;
        retval.is_aniso = false;
        return retval;
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso>)
    static SBxDFParams<Scalar> create(LightSample _sample, Aniso interaction, BxDFClampMode clamp = BCM_NONE)
    {
        this_t retval;
        retval.NdotV = clamp == BCM_ABS ? abs<Scalar>(interaction.NdotV) : 
                        clamp == BCM_MAX ? max<Scalar>(interaction.NdotV, 0.0) :
                                        interaction.NdotV;
        retval.uNdotV = interaction.NdotV;
        retval.NdotV2 = interaction.NdotV2;
        retval.NdotL = clamp == BCM_ABS ? abs<Scalar>(_sample.NdotL) :
                        clamp == BCM_MAX ? max<Scalar>(_sample.NdotL, 0.0) :
                                        _sample.NdotL;
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotL = _sample.VdotL;

        retval.is_aniso = true;
        retval.TdotL2 = _sample.TdotL * _sample.TdotL;
        retval.BdotL2 = _sample.BdotL * _sample.BdotL;
        retval.TdotV2 = interaction.TdotV * interaction.TdotV;
        retval.BdotV2 = interaction.BdotV * interaction.BdotV;
        return retval;
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    static this_t create(LightSample _sample, Iso interaction, Cache cache, BxDFClampMode clamp = BCM_NONE)
    {
        this_t retval;
        retval.NdotH = cache.NdotH;
        retval.NdotH2 = cache.NdotH2;
        retval.NdotV = clamp == BCM_ABS ? abs<Scalar>(interaction.NdotV) : 
                        clamp == BCM_MAX ? max<Scalar>(interaction.NdotV, 0.0) :
                                        interaction.NdotV;
        retval.uNdotV = interaction.NdotV;
        retval.NdotV2 = interaction.NdotV2;
        retval.NdotL = clamp == BCM_ABS ? abs<Scalar>(_sample.NdotL) :
                        clamp == BCM_MAX ? max<Scalar>(_sample.NdotL, 0.0) :
                                        _sample.NdotL;
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotH = cache.VdotH;
        retval.LdotH = cache.LdotH;
        retval.VdotL = _sample.VdotL;
        retval.is_aniso = false;
        return retval;
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    static SBxDFParams<Scalar> create(LightSample _sample, Aniso interaction, Cache cache, BxDFClampMode clamp = BCM_NONE)
    {
        this_t retval;
        retval.NdotH = cache.NdotH;
        retval.NdotH2 = cache.NdotH2;
        retval.NdotV = clamp == BCM_ABS ? abs<Scalar>(interaction.NdotV) : 
                        clamp == BCM_MAX ? max<Scalar>(interaction.NdotV, 0.0) :
                                        interaction.NdotV;
        retval.uNdotV = interaction.NdotV;
        retval.NdotV2 = interaction.NdotV2;
        retval.NdotL = clamp == BCM_ABS ? abs<Scalar>(_sample.NdotL) :
                        clamp == BCM_MAX ? max<Scalar>(_sample.NdotL, 0.0) :
                                        _sample.NdotL;
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotH = cache.VdotH;
        retval.LdotH = cache.LdotH;
        retval.VdotL = _sample.VdotL;

        retval.is_aniso = true;
        retval.TdotH2 = cache.TdotH * cache.TdotH;
        retval.BdotH2 = cache.BdotH * cache.BdotH;
        retval.TdotL2 = _sample.TdotL * _sample.TdotL;
        retval.BdotL2 = _sample.BdotL * _sample.BdotL;
        retval.TdotV2 = interaction.TdotV * interaction.TdotV;
        retval.BdotV2 = interaction.BdotV * interaction.BdotV;
        return retval;
    }

    Scalar getMaxNdotV() { return max<Scalar>(uNdotV, 0.0); }
    Scalar getAbsNdotV() { return abs<Scalar>(uNdotV); }

    Scalar getMaxNdotL() { return max<Scalar>(uNdotL, 0.0); }
    Scalar getAbsNdotL() { return abs<Scalar>(uNdotL); }

    // iso
    Scalar NdotH;
    Scalar NdotH2;
    Scalar NdotV;
    Scalar NdotV2;
    Scalar NdotL;
    Scalar NdotL2;
    Scalar VdotH;
    Scalar LdotH;
    Scalar VdotL;

    // aniso
    bool is_aniso;
    Scalar TdotH2;
    Scalar BdotH2;
    Scalar TdotL2;
    Scalar BdotL2;
    Scalar TdotV2;
    Scalar BdotV2;

    // original, unclamped
    Scalar uNdotL;
    Scalar uNdotV;
};

// fresnel stuff
namespace impl
{
template<typename T>
struct fresnel
{
    using scalar_t = typename scalar_type<T>::type;
    
    static T conductor(T eta, T etak, scalar_t cosTheta)
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

    static T dielectric(T orientedEta2, scalar_t absCosTheta)
    {
        const scalar_t sinTheta2 = 1.0 - absCosTheta * absCosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        const T t0 = nbl::hlsl::sqrt<T>(nbl::hlsl::max<T>((T)(orientedEta2) - sinTheta2, (T)(0.0)));
        const T rs = ((T)(absCosTheta) - t0) / ((T)(absCosTheta) + t0);

        const T t2 = orientedEta2 * absCosTheta;
        const T rp = (t0 - t2) / (t0 + t2);

        return (rs * rs + rp * rp) * 0.5f;
    }
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T fresnelSchlick(T F0, typename scalar_type<T>::type VdotH)
{
    T x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T fresnelConductor(T eta, T etak, typename scalar_type<T>::type cosTheta)
{
    return impl::fresnel<T>::conductor(eta, etak, cosTheta);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T fresnelDielectric_common(T eta, typename scalar_type<T>::type cosTheta)
{
    return impl::fresnel<T>::dielectric(eta, cosTheta);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T fresnelDielectricFrontFaceOnly(T eta, typename scalar_type<T>::type cosTheta)
{
    return impl::fresnel<T>::dielectric(eta * eta, cosTheta);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
T fresnelDielectric(T eta, typename scalar_type<T>::type cosTheta)
{
    T orientedEta, rcpOrientedEta;
    math::getOrientedEtas<T>(orientedEta, rcpOrientedEta, cosTheta, eta);
    return impl::fresnel<T>::dielectric(orientedEta * orientedEta, abs<typename scalar_type<T>::type>(cosTheta));
}

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
        return lerp<T>((singleInterfaceReflectance - doubleInterfaceReflectance) / ((T)(1.0) - doubleInterfaceReflectance) * 2.0f, (T)(1.0), doubleInterfaceReflectance > (T)(0.9999));
    }

    static scalar_t __call(scalar_t singleInterfaceReflectance) // TODO: check redundancy when lerp on line 980 works
    {
        const scalar_t doubleInterfaceReflectance = singleInterfaceReflectance * singleInterfaceReflectance;
        return doubleInterfaceReflectance > 0.9999 ? 1.0 : ((singleInterfaceReflectance - doubleInterfaceReflectance) / (1.0 - doubleInterfaceReflectance) * 2.0);
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
    // vector<bool,3> TIR = n < (T)1.0; // maybe make extent work in C++?
    T invdenum = lerp<T>((T)1.0, (T)1.0 / (n2 * n2 * ((T)554.33 - 380.7 * n)), n < (T)1.0);
    T num = n * lerp<T>((T)(0.1921156102251088), n * 298.25 - 261.38 * n2 + 138.43, n < (T)1.0);
    num += lerp<T>((T)(0.8078843897748912), (T)(-1.67), n < (T)1.0);
    return num * invdenum;
}

}
}
}

#endif
