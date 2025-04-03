// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/ieee754.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace bxdf
{

// returns unnormalized vector
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T computeUnnormalizedMicrofacetNormal(bool _refract, NBL_CONST_REF_ARG(T) V, NBL_CONST_REF_ARG(T) L, typename vector_traits<T>::scalar_type orientedEta)
{
    const typename vector_traits<T>::scalar_type etaFactor = (_refract ? orientedEta : 1.0);
    T tmpH = V + L * etaFactor;
    // ieee754::flipSign<T>(tmpH, _refract) doesn't work with vectors
    tmpH.x = ieee754::flipSign<typename vector_traits<T>::scalar_type>(tmpH.x, _refract);
    tmpH.y = ieee754::flipSign<typename vector_traits<T>::scalar_type>(tmpH.y, _refract);
    tmpH.z = ieee754::flipSign<typename vector_traits<T>::scalar_type>(tmpH.z, _refract);
    return tmpH;
}

// returns normalized vector, but NaN when result is length 0
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
T computeMicrofacetNormal(bool _refract, NBL_CONST_REF_ARG(T) V, NBL_CONST_REF_ARG(T) L, typename vector_traits<T>::scalar_type orientedEta)
{
    const T H = computeUnnormalizedMicrofacetNormal<T>(_refract,V,L,orientedEta);
    const typename vector_traits<T>::scalar_type unnormRcpLen = rsqrt<typename vector_traits<T>::scalar_type>(nbl::hlsl::dot<T>(H,H));
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_scalar_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_vector_v, typename T::vector3_type))
);
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
        bxdf::Reflect<vector3_type> r = bxdf::Reflect<vector3_type>::create(direction,N,directionDotN);
        retval.direction = r();
        return retval;
    }

    SBasic<T> refract(NBL_CONST_REF_ARG(vector3_type) N, scalar_type eta)
    {
        SBasic<T> retval;
        bxdf::Refract<vector3_type> r = bxdf::Refract<vector3_type>::create(direction,N,eta);
        retval.direction = r();
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Isotropic, typename T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
#undef normT
#undef iso
#undef aniso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo>)
struct SAnisotropic
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
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.V.getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.V.getDirection(), retval.B);

        return retval;
    }
    static SAnisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(isotropic_type) isotropic, NBL_CONST_REF_ARG(vector3_type) normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.N, normalizedT));
    }
    static SAnisotropic<RayDirInfo> create(NBL_CONST_REF_ARG(isotropic_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.N, T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.NdotV); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.N); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.N)); }

    isotropic_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};

}

template<typename InteractionT>
struct interaction_traits;

template<typename RayDirInfo>
struct interaction_traits<surface_interactions::SIsotropic<RayDirInfo> >
{
    NBL_CONSTEXPR_STATIC_INLINE bool is_aniso = false;
};

template<typename RayDirInfo>
struct interaction_traits<surface_interactions::SAnisotropic<RayDirInfo> >
{
    NBL_CONSTEXPR_STATIC_INLINE bool is_aniso = true;
};


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
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<typename T::ray_dir_info_type>(pV,iso)), ::nbl::hlsl::is_same_v, T)) // NOTE: temporarily commented out due to dxc bug https://github.com/microsoft/DirectXShaderCompiler/issues/7154
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<typename T::ray_dir_info_type>(pV,aniso)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getTangentSpaceL()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_type))
);
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

        retval.TdotL = nbl::hlsl::numeric_limits<scalar_type>::quiet_NaN;
        retval.BdotL = nbl::hlsl::numeric_limits<scalar_type>::quiet_NaN;
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
    // overloads for surface_interactions, NOTE: temporarily commented out due to dxc bug https://github.com/microsoft/DirectXShaderCompiler/issues/7154
    // template<class ObserverRayDirInfo>
    // static this_t create(NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(surface_interactions::SIsotropic<ObserverRayDirInfo>) interaction)
    // {
    //     const vector3_type V = interaction.V.getDirection();
    //     const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V,L);
    //     return create(L, VdotL, interaction.N);
    // }
    // template<class ObserverRayDirInfo>
    // static this_t create(NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(surface_interactions::SAnisotropic<ObserverRayDirInfo>) interaction)
    // {
    //     const vector3_type V = interaction.V.getDirection();
    //     const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V,L);
    //     return create(L,VdotL,interaction.T,interaction.B,interaction.N);
    // }
    //
    vector3_type getTangentSpaceL() NBL_CONST_MEMBER_FUNC
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_type))
);
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
        H = computeMicrofacetNormal<vector3_type>(transmitted,V,L,orientedEta);
        retval.NdotH = nbl::hlsl::dot<vector3_type>(N, H);

        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) )
        const bool valid = !transmitted || (VdotL <= -min(orientedEta, rcpOrientedEta) && retval.NdotH > nbl::hlsl::numeric_limits<scalar_type>::min);
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

        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(NdotV, eta);

        const vector3_type V = interaction.V.getDirection();
        const vector3_type L = _sample.L.direction;
        const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V, L);
        return compute(retval,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta.value,orientedEta.rcp,H);
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_type))
);
#undef b0
#undef V
#undef _sample
#undef pNdotL
#undef aniso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename U NBL_PRIMARY_REQUIRES(is_scalar_v<U>)
struct SAnisotropicMicrofacetCache
{
    using this_t = SAnisotropicMicrofacetCache<U>;
    using scalar_type = U;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    using ray_dir_info_type = ray_dir_info::SBasic<scalar_type>;
    using anisotropic_type = surface_interactions::SAnisotropic<ray_dir_info_type>;
    using isocache_type = SIsotropicMicrofacetCache<U>;
    using sample_type = SLightSample<ray_dir_info_type>;

    // always valid by construction
    static this_t create(NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, NBL_CONST_REF_ARG(vector3_type) tangentSpaceH)
    {
        this_t retval;

        retval.iso_cache.VdotH = nbl::hlsl::dot<vector3_type>(tangentSpaceV,tangentSpaceH);
        retval.iso_cache.LdotH = retval.iso_cache.VdotH;
        retval.iso_cache.NdotH = tangentSpaceH.z;
        retval.iso_cache.NdotH2 = retval.iso_cache.NdotH * retval.iso_cache.NdotH;
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
            const scalar_type VdotH = retval.iso_cache.VdotH;
            retval.iso_cache.LdotH = transmitted ? refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2) : VdotH;
        }

        return retval;
    }
    // always valid because its specialized for the reflective case
    static this_t createForReflection(NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, NBL_CONST_REF_ARG(vector3_type) tangentSpaceL, const scalar_type VdotL)
    {
        this_t retval;

        scalar_type LplusV_rcpLen;
        retval.iso_cache = SIsotropicMicrofacetCache<U>::createForReflection(tangentSpaceV.z, tangentSpaceL.z, VdotL, LplusV_rcpLen);
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
        const bool valid = isocache_type::compute(retval.iso_cache,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
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
        const bool valid = isocache_type::template compute<ObserverRayDirInfo, IncomingRayDirInfo>(retval.iso_cache,interaction.isotropic,_sample,eta,H);
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(interaction.T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(interaction.B,H);
        }
        return valid;
    }

    isocache_type iso_cache;
    scalar_type TdotH;
    scalar_type BdotH;
};

template<typename CacheT>
struct microfacet_cache_traits;

template<typename U>
struct microfacet_cache_traits<SIsotropicMicrofacetCache<U> >
{
    NBL_CONSTEXPR_STATIC_INLINE bool is_aniso = false;
};

template<typename U>
struct microfacet_cache_traits<SAnisotropicMicrofacetCache<U> >
{
    NBL_CONSTEXPR_STATIC_INLINE bool is_aniso = true;
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_scalar_v, F))
);
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Sample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(spectral_of, typename T::spectral_type, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_type))
);
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Sample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(spectral_of, typename T::spectral_type, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(IsotropicMicrofacetCache, typename T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(AnisotropicMicrofacetCache, typename T::anisocache_type))
);
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

namespace impl
{
// this is to substitute the lack of compile-time `if constexpr` on HLSL
template<class LightSample, class Interaction, typename T, bool is_aniso>
struct __extract_aniso_vars;

template<class LightSample, class Interaction, typename T>
struct __extract_aniso_vars<LightSample, Interaction, T, false>
{
    static __extract_aniso_vars<LightSample, Interaction, T, false> create(NBL_CONST_REF_ARG(LightSample) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        __extract_aniso_vars<LightSample, Interaction, T, false> retval;
        retval.NdotV = interaction.NdotV;
        retval.NdotV2 = interaction.NdotV2;
        return retval;
    }

    T NdotV;
    T NdotV2;
    T TdotL2;
    T BdotL2;
    T TdotV2;
    T BdotV2;
};

template<class LightSample, class Interaction, typename T>
struct __extract_aniso_vars<LightSample, Interaction, T, true>
{
    static __extract_aniso_vars<LightSample, Interaction, T, true> create(NBL_CONST_REF_ARG(LightSample) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        __extract_aniso_vars<LightSample, Interaction, T, true> retval;
        retval.NdotV = interaction.isotropic.NdotV;
        retval.NdotV2 = interaction.isotropic.NdotV2;
        retval.TdotL2 = _sample.TdotL * _sample.TdotL;
        retval.BdotL2 = _sample.BdotL * _sample.BdotL;
        retval.TdotV2 = interaction.TdotV * interaction.TdotV;
        retval.BdotV2 = interaction.BdotV * interaction.BdotV;
        return retval;
    }

    T NdotV;
    T NdotV2;
    T TdotL2;
    T BdotL2;
    T TdotV2;
    T BdotV2;
};

template<class Cache, typename T, bool is_aniso>
struct __extract_aniso_vars2;

template<class Cache, typename T>
struct __extract_aniso_vars2<Cache, T, false>
{
    static __extract_aniso_vars2<Cache, T, false> create(NBL_CONST_REF_ARG(Cache) cache)
    {
        __extract_aniso_vars2<Cache, T, false> retval;
        retval.NdotH = cache.NdotH;
        retval.NdotH2 = cache.NdotH2;
        retval.VdotH = cache.VdotH;
        retval.LdotH = cache.LdotH;
        return retval;
    }

    T NdotH;
    T NdotH2;
    T VdotH;
    T LdotH;
    T TdotH2;
    T BdotH2;
};

template<class Cache, typename T>
struct __extract_aniso_vars2<Cache, T, true>
{
    static __extract_aniso_vars2<Cache, T, true> create(NBL_CONST_REF_ARG(Cache) cache)
    {
        __extract_aniso_vars2<Cache, T, true> retval;
        retval.NdotH = cache.iso_cache.NdotH;
        retval.NdotH2 = cache.iso_cache.NdotH2;
        retval.VdotH = cache.iso_cache.VdotH;
        retval.LdotH = cache.iso_cache.LdotH;
        retval.TdotH2 = cache.TdotH * cache.TdotH;
        retval.BdotH2 = cache.BdotH * cache.BdotH;
        return retval;
    }

    T NdotH;
    T NdotH2;
    T VdotH;
    T LdotH;
    T TdotH2;
    T BdotH2;
};
}

// unified param struct for calls to BxDF::eval, BxDF::pdf, BxDF::quotient_and_pdf
template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFParams
{
    using this_t = SBxDFParams<Scalar>;

    template<class LightSample, class Interaction NBL_FUNC_REQUIRES(Sample<LightSample> && (surface_interactions::Isotropic<Interaction> || surface_interactions::Anisotropic<Interaction>))    // maybe put template in struct vs function?
    static this_t create(NBL_CONST_REF_ARG(LightSample) _sample, NBL_CONST_REF_ARG(Interaction) interaction, BxDFClampMode _clamp)
    {
        impl::__extract_aniso_vars<LightSample, Interaction, Scalar, interaction_traits<Interaction>::is_aniso> vars = impl::__extract_aniso_vars<LightSample, Interaction, Scalar, interaction_traits<Interaction>::is_aniso>::create(_sample, interaction);

        this_t retval;
        retval.NdotV = math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, vars.NdotV, 0.0);
        retval.uNdotV = vars.NdotV;
        retval.NdotV2 = vars.NdotV2;
        retval.NdotL = math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, _sample.NdotL, 0.0);
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotL = _sample.VdotL;

        retval.is_aniso = interaction_traits<Interaction>::is_aniso;
        retval.TdotL2 = vars.TdotL2;
        retval.BdotL2 = vars.BdotL2;
        retval.TdotV2 = vars.TdotV2;
        retval.BdotV2 = vars.BdotV2;
        return retval;
    }

    template<class LightSample, class Interaction, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && (surface_interactions::Isotropic<Interaction> || surface_interactions::Anisotropic<Interaction>) && (IsotropicMicrofacetCache<Cache> || AnisotropicMicrofacetCache<Cache>))
    static this_t create(NBL_CONST_REF_ARG(LightSample) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(Cache) cache, BxDFClampMode _clamp)
    {
        impl::__extract_aniso_vars<LightSample, Interaction, Scalar, interaction_traits<Interaction>::is_aniso> vars = impl::__extract_aniso_vars<LightSample, Interaction, Scalar, interaction_traits<Interaction>::is_aniso>::create(_sample, interaction);
        impl::__extract_aniso_vars2<Cache, Scalar, interaction_traits<Interaction>::is_aniso> vars2 = impl::__extract_aniso_vars2<Cache, Scalar, interaction_traits<Interaction>::is_aniso>::create(cache);

        this_t retval;
        retval.NdotH = vars2.NdotH;
        retval.NdotH2 = vars2.NdotH2;
        retval.NdotV = math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, vars.NdotV, 0.0);
        retval.uNdotV = vars.NdotV;
        retval.NdotV2 = vars.NdotV2;
        retval.NdotL = math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, _sample.NdotL, 0.0);
        retval.uNdotL = _sample.NdotL;
        retval.NdotL2 = _sample.NdotL2;
        retval.VdotH = vars2.VdotH;
        retval.LdotH = vars2.LdotH;
        retval.VdotL = _sample.VdotL;

        retval.is_aniso = interaction_traits<Interaction>::is_aniso;
        retval.TdotL2 = vars.TdotL2;
        retval.BdotL2 = vars.BdotL2;
        retval.TdotV2 = vars.TdotV2;
        retval.BdotV2 = vars.BdotV2;
        retval.TdotH2 = vars2.TdotH2;
        retval.BdotH2 = vars2.BdotH2;
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

// unified param struct for calls to BxDF::create
template<typename Scalar, typename Spectrum NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFCreationParams
{
    bool is_aniso;
    vector<Scalar, 2> A;    // roughness
    Spectrum ior0;          // source ior
    Spectrum ior1;          // destination ior
    Scalar eta;             // in most cases, eta will be calculated from ior0 and ior1; see monochromeEta in pathtracer.hlsl
    Spectrum eta2;
    Spectrum luminosityContributionHint;
};

}
}
}

#endif
