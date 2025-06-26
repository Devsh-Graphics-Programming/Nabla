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
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace bxdf
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<T>)
struct ComputeMicrofacetNormal
{
    using vector_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    static ComputeMicrofacetNormal<T> create(NBL_CONST_REF_ARG(vector_type) V, NBL_CONST_REF_ARG(vector_type) L, NBL_CONST_REF_ARG(vector_type) H, scalar_type eta)
    {
        ComputeMicrofacetNormal<T> retval;
        retval.V = V;
        retval.L = L;
        retval.orientedEta = fresnel::OrientedEtas<scalar_type>::create(hlsl::dot<vector_type>(V, H), eta);
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
        const scalar_type etaFactor = hlsl::mix(scalar_type(1.0), orientedEta.value, _refract);
        vector_type tmpH = V + L * etaFactor;
        tmpH = ieee754::flipSign<vector_type>(tmpH, _refract);
        return tmpH;
    }

    // returns normalized vector, but NaN when result is length 0
    vector_type normalized(const bool _refract)
    {
        const vector_type H = unnormalized(_refract,V,L,orientedEta);
        return hlsl::normalize<vector_type>(H);
    }

    // if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
    static bool isTransmissionPath(float NdotV, float NdotL)
    {
        return bool((bit_cast<uint32_t>(NdotV) ^ bit_cast<uint32_t>(NdotL)) & 0x80000000u);
    }

    vector_type V;
    vector_type L;
    fresnel::OrientedEtas<scalar_type> orientedEta;
};


namespace ray_dir_info
{

#define NBL_CONCEPT_NAME Basic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (rdirinfo, T)
#define NBL_CONCEPT_PARAM_1 (N, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_2 (dirDotN, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (m, typename T::matrix3x3_type)
#define NBL_CONCEPT_PARAM_4 (rfl, Reflect<typename T::scalar_type>)
#define NBL_CONCEPT_PARAM_5 (rfr, Refract<typename T::scalar_type>)
NBL_CONCEPT_BEGIN(6)
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define N NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define dirDotN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define m NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define rfl NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define rfr NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.getDirection()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transmit()), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflect(rfl)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.refract(rfr)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transform(m)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_scalar_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_vector_v, typename T::vector3_type))
);
#undef rfr
#undef rfl
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

    SBasic<T> reflect(NBL_CONST_REF_ARG(Reflect<scalar_type>) r)
    {
        SBasic<T> retval;
        retval.direction = r();
        return retval;
    }

    SBasic<T> refract(NBL_CONST_REF_ARG(Refract<scalar_type>) r)
    {
        SBasic<T> retval;
        retval.direction = r();
        return retval;
    }

    // WARNING: matrix must be orthonormal
    SBasic<T> transform(NBL_CONST_REF_ARG(matrix3x3_type) m) NBL_CONST_MEMBER_FUNC
    {
        matrix3x3_type m_T = nbl::hlsl::transpose<matrix3x3_type>(m);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[1])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[2])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[1], m_T[2])) < 1e-5);

        SBasic<T> retval;
        retval.direction = nbl::hlsl::mul<matrix3x3_type,vector3_type>(m, direction);
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
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getV()), ::nbl::hlsl::is_same_v, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getN()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getNdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getNdotV2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
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

    RayDirInfo getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV() NBL_CONST_MEMBER_FUNC { return NdotV; }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    RayDirInfo V;
    vector3_type N;
    scalar_type NdotV;
    scalar_type NdotV2;
};

#define NBL_CONCEPT_NAME Anisotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (aniso, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (normT, typename T::vector3_type)
NBL_CONCEPT_BEGIN(3)
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normT NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getT()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getB()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getBdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso,normT,normT)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTangentSpaceV()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getToTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getFromTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Isotropic, typename T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
#undef normT
#undef iso
#undef aniso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class IsotropicInteraction NBL_PRIMARY_REQUIRES(Isotropic<IsotropicInteraction>)
struct SAnisotropic
{
    using this_t = SAnisotropic<IsotropicInteraction>;
    using isotropic_interaction_type = IsotropicInteraction;
    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(
        NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic,
        NBL_CONST_REF_ARG(vector3_type) normalizedT,
        NBL_CONST_REF_ARG(vector3_type) normalizedB
    )
    {
        this_t retval;
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.B);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, NBL_CONST_REF_ARG(vector3_type) normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, NBL_CONST_REF_ARG(vector3_type) normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.N; }
    scalar_type getNdotV() NBL_CONST_MEMBER_FUNC { return isotropic.NdotV; }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.NdotV2; }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.NdotV); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.N); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.N)); }

    isotropic_interaction_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};

}


#define NBL_CONCEPT_NAME LightSample
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (_sample, T)
#define NBL_CONCEPT_PARAM_1 (inter, surface_interactions::SIsotropic<typename T::ray_dir_info_type>)
#define NBL_CONCEPT_PARAM_2 (rdirinfo, typename T::ray_dir_info_type)
#define NBL_CONCEPT_PARAM_3 (pV, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_4 (frame, typename T::matrix3x3_type)
#define NBL_CONCEPT_PARAM_5 (pVdotL, typename T::scalar_type)
NBL_CONCEPT_BEGIN(6)
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define inter NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define pV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define frame NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define pVdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getL()), ::nbl::hlsl::is_same_v, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getVdotL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getTdotL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getBdotL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getNdotL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getNdotL2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createFromTangentSpace(pV,rdirinfo,frame)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(rdirinfo,pVdotL,pV,pV,pV)), ::nbl::hlsl::is_same_v, T))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<surface_interactions::SIsotropic<typename T::ray_dir_info_type> >(pV,inter)), ::nbl::hlsl::is_same_v, T)) // NOTE: temporarily commented out due to dxc bug https://github.com/microsoft/DirectXShaderCompiler/issues/7154
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((_sample.getTangentSpaceL()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
#undef pVdotL
#undef frame
#undef pV
#undef rdirinfo
#undef inter
#undef _sample
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo>)
struct SLightSample
{
    using this_t = SLightSample<RayDirInfo>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    static this_t createFromTangentSpace(
        NBL_CONST_REF_ARG(vector3_type) tangentSpaceV,
        NBL_CONST_REF_ARG(ray_dir_info_type) tangentSpaceL,
        NBL_CONST_REF_ARG(matrix3x3_type) tangentFrame
    )
    {
        this_t retval;

        const vector3_type tsL = tangentSpaceL.getDirection();
        retval.L = tangentSpaceL.transform(tangentFrame);
        retval.VdotL = nbl::hlsl::dot<vector3_type>(tangentSpaceV, tsL);

        retval.TdotL = tsL.x;
        retval.BdotL = tsL.y;
        retval.NdotL = tsL.z;
        retval.NdotL2 = retval.NdotL*retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) L, const scalar_type VdotL, NBL_CONST_REF_ARG(vector3_type) N)
    {
        this_t retval;

        retval.L = L;
        retval.VdotL = VdotL;

        retval.TdotL = nbl::hlsl::numeric_limits<scalar_type>::quiet_NaN;
        retval.BdotL = nbl::hlsl::numeric_limits<scalar_type>::quiet_NaN;
        retval.NdotL = nbl::hlsl::dot<vector3_type>(N,L.getDirection());
        retval.NdotL2 = retval.NdotL * retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) L, const scalar_type VdotL, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N)
    {
        this_t retval = create(L,VdotL,N);

        retval.TdotL = nbl::hlsl::dot<vector3_type>(T,L.getDirection());
        retval.BdotL = nbl::hlsl::dot<vector3_type>(B,L.getDirection());

        return retval;
    }

    template<class SurfaceInteraction NBL_FUNC_REQUIRES(surface_interactions::Isotropic<SurfaceInteraction>)
    static this_t create(NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(SurfaceInteraction) interaction)
    {
        const vector3_type V = interaction.V.getDirection();
        const scalar_type VdotL = nbl::hlsl::dot<vector3_type>(V,L);
        this_t retval;
        if (surface_interactions::Anisotropic<SurfaceInteraction>)  // TODO use NBL_IF_CONSTEXPR when pr #860 is merged
            retval = create(L,VdotL,interaction.T,interaction.B,interaction.N);
        else
            retval = create(L, VdotL, interaction.N);
        return retval;
    }
    
    vector3_type getTangentSpaceL() NBL_CONST_MEMBER_FUNC
    {
        return vector3_type(TdotL, BdotL, NdotL);
    }

    ray_dir_info_type getL() NBL_CONST_MEMBER_FUNC { return L; }
    scalar_type getVdotL() NBL_CONST_MEMBER_FUNC { return VdotL; }
    scalar_type getTdotL() NBL_CONST_MEMBER_FUNC { return TdotL; }
    scalar_type getBdotL() NBL_CONST_MEMBER_FUNC { return BdotL; }
    scalar_type getNdotL() NBL_CONST_MEMBER_FUNC { return NdotL; }
    scalar_type getNdotL2() NBL_CONST_MEMBER_FUNC { return NdotL2; }


    RayDirInfo L;
    scalar_type VdotL;

    scalar_type TdotL;
    scalar_type BdotL;
    scalar_type NdotL;
    scalar_type NdotL2;
};


#define NBL_CONCEPT_NAME ReadableIsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (pNdotV, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_2 (b0, bool)
NBL_CONCEPT_BEGIN(3)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define pNdotV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getVdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getLdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getNdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getNdotH2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.isValidVNDFMicrofacet(b0,b0,pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, bool))
);
#undef b0
#undef pNdotV
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME CreatableIsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (iso, surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> >)
#define NBL_CONCEPT_PARAM_2 (pNdotV, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, SLightSample<ray_dir_info::SBasic<typename T::scalar_type> >)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (eta, fresnel::OrientedEtas<typename T::scalar_type>)
NBL_CONCEPT_BEGIN(6)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define eta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(pNdotV,pNdotV,pNdotV)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(iso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,V,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(iso,_sample,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ReadableIsotropicMicrofacetCache, T))
);
#undef eta
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
    template<class IsotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Isotropic<IsotropicInteraction> && LightSample<LS>)
    static this_t createForReflection(
        NBL_CONST_REF_ARG(IsotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample)
    {
        return createForReflection(interaction.getNdotV(), _sample.getNdotL(), _sample.getVdotL());
    }

    // transmissive cases need to be checked if the path is valid before usage
    static this_t create(
        NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) L, NBL_CONST_REF_ARG(vector3_type) N,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEtas, NBL_REF_ARG(vector3_type) H)
    {
        this_t retval;
        const scalar_type NdotV = hlsl::dot<vector3_type>(N, V);
        const scalar_type NdotL = hlsl::dot<vector3_type>(N, L);
        const scalar_type VdotL = hlsl::dot<vector3_type>(V, L);
        const bool transmitted = ComputeMicrofacetNormal<vector3_type>::isTransmissionPath(NdotV,NdotL);

        ComputeMicrofacetNormal<vector3_type> computeMicrofacetNormal = ComputeMicrofacetNormal<vector3_type>::create(V,L,N,1.0);
        computeMicrofacetNormal.orientedEta.value = orientedEtas.value;
        computeMicrofacetNormal.orientedEta.rcp = orientedEtas.rcp;
        H = computeMicrofacetNormal.normalized(transmitted);
        retval.NdotH = hlsl::dot<vector3_type>(N, H);

        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) )
        const bool valid = !transmitted || (VdotL <= -hlsl::min(orientedEtas.value, orientedEtas.rcp) && retval.NdotH >= nbl::hlsl::numeric_limits<scalar_type>::min);
        if (valid)
        {
            retval.VdotH = hlsl::dot<vector3_type>(V,H);
            retval.LdotH = hlsl::dot<vector3_type>(L,H);
            retval.NdotH2 = retval.NdotH * retval.NdotH;
        }
        else
            retval.NdotH = -1.0;
        return retval;
    }

    template<class IsotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Isotropic<IsotropicInteraction> && LightSample<LS>)
    static this_t create(
        NBL_CONST_REF_ARG(IsotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEtas, NBL_REF_ARG(vector3_type) H)
    {
        const vector3_type V = interaction.getV().getDirection();
        const vector3_type L = _sample.getL().getDirection();

        return create(V,L,interaction.getN(),orientedEtas,H);
    }

    bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const scalar_type VdotL, const scalar_type eta, const scalar_type rcp_eta)
    {
        return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -hlsl::min(eta, rcp_eta)));
    }

    scalar_type getVdotH() NBL_CONST_MEMBER_FUNC
    {
        assert(VdotH >= scalar_type(0.0));
        return VdotH; 
    }

    scalar_type getLdotH() NBL_CONST_MEMBER_FUNC { return LdotH; }
    scalar_type getNdotH() NBL_CONST_MEMBER_FUNC { return NdotH; }
    scalar_type getNdotH2() NBL_CONST_MEMBER_FUNC { return NdotH2; }

    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type NdotH;
    scalar_type NdotH2;
};


#define NBL_CONCEPT_NAME AnisotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (aniso, surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> > >)
#define NBL_CONCEPT_PARAM_2 (pNdotL, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, SLightSample<ray_dir_info::SBasic<typename T::scalar_type> >)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (b0, bool)
#define NBL_CONCEPT_PARAM_6 (eta, fresnel::OrientedEtas<typename T::scalar_type>)
NBL_CONCEPT_BEGIN(7)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define eta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getTdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getBdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,b0,pNdotL,pNdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(V,V,pNdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> > >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(aniso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,V,V,V,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type> > >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(aniso,_sample,eta)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(CreatableIsotropicMicrofacetCache, typename T::isocache_type))
);
#undef eta
#undef b0
#undef V
#undef _sample
#undef pNdotL
#undef aniso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class IsoCache NBL_PRIMARY_REQUIRES(CreatableIsotropicMicrofacetCache<IsoCache>)
struct SAnisotropicMicrofacetCache
{
    using this_t = SAnisotropicMicrofacetCache<IsoCache>;
    using isocache_type = IsoCache;
    using scalar_type = typename IsoCache::scalar_type;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    // always valid by construction
    static this_t create(NBL_CONST_REF_ARG(vector3_type) tangentSpaceV, NBL_CONST_REF_ARG(vector3_type) tangentSpaceH)
    {
        this_t retval;

        retval.iso_cache.VdotH = nbl::hlsl::dot<vector3_type>(tangentSpaceV,tangentSpaceH);
        retval.iso_cache.LdotH = retval.iso_cache.getVdotH();
        retval.iso_cache.NdotH = tangentSpaceH.z;
        retval.iso_cache.NdotH2 = retval.iso_cache.getNdotH() * retval.iso_cache.getNdotH();
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
        retval.iso_cache = isocache_type::createForReflection(tangentSpaceV.z, tangentSpaceL.z, VdotL, LplusV_rcpLen);
        retval.TdotH = (tangentSpaceV.x + tangentSpaceL.x) * LplusV_rcpLen;
        retval.BdotH = (tangentSpaceV.y + tangentSpaceL.y) * LplusV_rcpLen;

        return retval;
    }
    template<class AnisotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Anisotropic<AnisotropicInteraction> && LightSample<LS>)
    static this_t createForReflection(
        NBL_CONST_REF_ARG(AnisotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample)
    {
        return createForReflection(interaction.getTangentSpaceV(), _sample.getTangentSpaceL(), _sample.getVdotL());
    }
    // transmissive cases need to be checked if the path is valid before usage
    static this_t create(
        NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) L,
        NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEtas, NBL_REF_ARG(vector3_type) H
    )
    {
        this_t retval;
        retval.iso_cache = isocache_type::create(V,L,N,orientedEtas,H);
        const bool valid = retval.iso_cache.NdotH >= 0.0;
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(B,H);
        }
        return retval;
    }
    template<class AnisotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Anisotropic<AnisotropicInteraction> && LightSample<LS>)
    static this_t create(
        NBL_CONST_REF_ARG(AnisotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEtas
    )
    {
        this_t retval;
        vector3_type H;
        retval.iso_cache = isocache_type::template create<AnisotropicInteraction::isotropic_interaction_type, LS>(interaction.isotropic,_sample,orientedEtas,H);
        const bool valid = retval.iso_cache.NdotH >= 0.0;
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(interaction.getT(),H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(interaction.getB(),H);
        }
        return retval;
    }

    bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const scalar_type VdotL, const scalar_type eta, const scalar_type rcp_eta)
    {
        return iso_cache.isValidVNDFMicrofacet(is_bsdf, transmission, VdotL, eta, rcp_eta);
    }

    scalar_type getVdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.VdotH; }
    scalar_type getLdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.LdotH; }
    scalar_type getNdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.NdotH; }
    scalar_type getNdotH2() NBL_CONST_MEMBER_FUNC { return iso_cache.NdotH2; }

    scalar_type getTdotH() NBL_CONST_MEMBER_FUNC { return TdotH; }
    scalar_type getBdotH() NBL_CONST_MEMBER_FUNC { return BdotH; }

    isocache_type iso_cache;
    scalar_type TdotH;
    scalar_type BdotH;
};


#define NBL_CONCEPT_NAME BxDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (spec, typename T::spectral_type)
#define NBL_CONCEPT_PARAM_2 (pdf, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_4 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_5 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_6 (param_iso, typename T::params_isotropic_t)
#define NBL_CONCEPT_PARAM_7 (param_aniso, typename T::params_anisotropic_t)
#define NBL_CONCEPT_PARAM_8 (u, vector<typename T::scalar_type, 2>)
NBL_CONCEPT_BEGIN(9)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define spec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define param_iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define param_aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::params_isotropic_t))
    ((NBL_CONCEPT_REQ_TYPE)(T::params_anisotropic_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param_iso)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param_aniso)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(iso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param_iso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param_aniso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(LightSample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(sampling::Spectral, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_interaction_type))
);
#undef u
#undef param_aniso
#undef param_iso
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
#define NBL_CONCEPT_PARAM_4 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_5 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_6 (isocache, typename T::isocache_type)
#define NBL_CONCEPT_PARAM_7 (anisocache, typename T::anisocache_type)
#define NBL_CONCEPT_PARAM_8 (param_iso, typename T::params_isotropic_t)
#define NBL_CONCEPT_PARAM_9 (param_aniso, typename T::params_anisotropic_t)
#define NBL_CONCEPT_PARAM_10 (u, vector<typename T::scalar_type, 3>)
NBL_CONCEPT_BEGIN(11)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define spec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define isocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define anisocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
#define param_iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_8
#define param_aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_9
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_10
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::params_isotropic_t))
    ((NBL_CONCEPT_REQ_TYPE)(T::params_anisotropic_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param_iso)), ::nbl::hlsl::is_same_v, T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(param_aniso)), ::nbl::hlsl::is_same_v, T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(iso,u,isocache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,u,anisocache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
    //((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.template pdf<LS,I>(_sample,iso)), ::nbl::hlsl::is_scalar_v))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param_iso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(param_aniso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(LightSample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(sampling::Spectral, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(CreatableIsotropicMicrofacetCache, typename T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(AnisotropicMicrofacetCache, typename T::anisocache_type))
);
#undef u
#undef param_aniso
#undef param_iso
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
