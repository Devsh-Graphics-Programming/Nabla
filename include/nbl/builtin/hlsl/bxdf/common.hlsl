// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/ieee754.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"
// #include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
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

namespace ray_dir_info
{

#define NBL_CONCEPT_NAME Basic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (rdirinfo, T)
#define NBL_CONCEPT_PARAM_1 (v, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_2 (rcpEta, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (m, typename T::matrix3x3_type)
#define NBL_CONCEPT_PARAM_4 (rfl, Reflect<typename T::scalar_type>)
#define NBL_CONCEPT_PARAM_5 (rfr, Refract<typename T::scalar_type>)
#define NBL_CONCEPT_PARAM_6 (t, bool)
#define NBL_CONCEPT_PARAM_7 (rr, ReflectRefract<typename T::scalar_type>)
NBL_CONCEPT_BEGIN(8)
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define v NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define rcpEta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define m NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define rfl NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define rfr NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define t NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define rr NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR)(rdirinfo.setDirection(v)))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.getDirection()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transmit()), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflect(rfl)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.refract(rfr, rcpEta)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflectTransmit(rfl, t)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.reflectRefract(rr, t, rcpEta)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.transform(m)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((rdirinfo.isValid()), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR)(rdirinfo.makeInvalid()))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_scalar_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(is_vector_v, typename T::vector3_type))
);
#undef rr
#undef t
#undef rfr
#undef rfl
#undef m
#undef rcpEta
#undef v
#undef rdirinfo
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T>
struct SBasic
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;
    using matrix3x3_type = matrix<T, 3, 3>;

    void setDirection(const vector3_type v) { direction = v; }
    vector3_type getDirection() NBL_CONST_MEMBER_FUNC { return direction; }

    SBasic<T> transmit() NBL_CONST_MEMBER_FUNC
    {
        SBasic<T> retval;
        retval.direction = -direction;
        return retval;
    }

    template<typename R=Reflect<scalar_type> >
    SBasic<T> reflect(NBL_CONST_REF_ARG(R) r) NBL_CONST_MEMBER_FUNC
    {
        SBasic<T> retval;
        retval.direction = r();
        return retval;
    }

    template<typename R=Refract<scalar_type> >
    SBasic<T> refract(NBL_CONST_REF_ARG(R) r, scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        SBasic<T> retval;
        retval.direction = r(rcpOrientedEta);
        return retval;
    }

    template<typename R=Reflect<scalar_type> >
    SBasic<T> reflectTransmit(NBL_CONST_REF_ARG(R) r, bool transmitted) NBL_CONST_MEMBER_FUNC
    {
        SBasic<T> retval;
        retval.direction = hlsl::mix(r(), -direction, transmitted);
        return retval;
    }

    template<typename R=ReflectRefract<scalar_type> >
    SBasic<T> reflectRefract(NBL_CONST_REF_ARG(R) rr, bool transmitted, scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
    {
        SBasic<T> retval;
        retval.direction = rr(transmitted, rcpOrientedEta);
        return retval;
    }

    // WARNING: matrix must be orthonormal
    SBasic<T> transform(const matrix3x3_type m) NBL_CONST_MEMBER_FUNC
    {
        matrix3x3_type m_T = nbl::hlsl::transpose<matrix3x3_type>(m);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[1])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[0], m_T[2])) < 1e-5);
        assert(nbl::hlsl::abs<scalar_type>(nbl::hlsl::dot<vector3_type>(m_T[1], m_T[2])) < 1e-5);

        SBasic<T> retval;
        retval.direction = nbl::hlsl::mul<matrix3x3_type,vector3_type>(m, direction);
        return retval;
    }

    void makeInvalid()
    {
        direction = vector3_type(0,0,0);
    }

    bool isValid() NBL_CONST_MEMBER_FUNC { return hlsl::any<vector<bool, 3> >(hlsl::glsl::notEqual(direction, hlsl::promote<vector3_type>(0.0))); }

    vector3_type direction;
};
// more to come!

}

enum BxDFClampMode : uint16_t
{
    BCM_NONE = 0,
    BCM_MAX,
    BCM_ABS
};

enum PathOrigin : uint16_t
{
    PO_SENSOR = 0,
    PO_LIGHT
};

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointScalar<T>)
T conditionalAbsOrMax(T x, BxDFClampMode _clamp)
{
    return hlsl::mix(math::conditionalAbsOrMax<T>(_clamp == BxDFClampMode::BCM_ABS, x, 0.0), x, _clamp == BxDFClampMode::BCM_NONE);
}

namespace surface_interactions
{

#define NBL_CONCEPT_NAME Isotropic
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (iso, T)
#define NBL_CONCEPT_PARAM_1 (normV, typename T::ray_dir_info_type)
#define NBL_CONCEPT_PARAM_2 (normN, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_3 (clampMode, BxDFClampMode)
NBL_CONCEPT_BEGIN(4)
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define normV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define normN NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define clampMode NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getV()), ::nbl::hlsl::is_same_v, typename T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getN()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getNdotV(clampMode)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getNdotV2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getPathOrigin()), ::nbl::hlsl::is_same_v, PathOrigin))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((iso.getLuminosityContributionHint()), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(normV,normN)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
#undef clampMode
#undef normN
#undef normV
#undef iso
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<class RayDirInfo, class Spectrum NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SIsotropic
{
    using this_t = SIsotropic<RayDirInfo, Spectrum>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using spectral_type = vector3_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, const vector3_type normalizedN)
    {
        this_t retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = nbl::hlsl::dot<vector3_type>(retval.N, retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;
        retval.luminosityContributionHint = hlsl::promote<spectral_type>(1.0);

        return retval;
    }

    RayDirInfo getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return PathOrigin::PO_SENSOR; }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return luminosityContributionHint; }

    RayDirInfo V;
    vector3_type N;
    scalar_type NdotV;
    scalar_type NdotV2;

    spectral_type luminosityContributionHint;
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
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Isotropic, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getT()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getB()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTdotV2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getBdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getBdotV2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(iso,normT,normT)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getTangentSpaceV()), ::nbl::hlsl::is_same_v, typename T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getToTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.getFromTangentSpace()), ::nbl::hlsl::is_same_v, typename T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(Isotropic, typename T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((aniso.isotropic), ::nbl::hlsl::is_same_v, typename T::isotropic_interaction_type))
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
    using spectral_type = typename isotropic_interaction_type::spectral_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(
        NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic,
        const vector3_type normalizedT,
        const vector3_type normalizedB
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
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.getV(); }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.getN(); }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV(_clamp); }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV2(); }
    PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return isotropic.getPathOrigin(); }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return isotropic.getLuminosityContributionHint(); }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getTdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotV(); return t*t; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }
    scalar_type getBdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotV(); return t*t; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.getNdotV()); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.getN()); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.getN())); }

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
#define NBL_CONCEPT_PARAM_1 (inter, surface_interactions::SIsotropic<typename T::ray_dir_info_type, typename T::vector3_type>)
#define NBL_CONCEPT_PARAM_2 (rdirinfo, typename T::ray_dir_info_type)
#define NBL_CONCEPT_PARAM_3 (pV, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_4 (frame, typename T::matrix3x3_type)
#define NBL_CONCEPT_PARAM_5 (clampMode, BxDFClampMode)
#define NBL_CONCEPT_PARAM_6 (pNdotL, typename T::scalar_type)
NBL_CONCEPT_BEGIN(7)
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define inter NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define rdirinfo NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define pV NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define frame NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define clampMode NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
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
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createInvalid()), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ray_dir_info::Basic, typename T::ray_dir_info_type))
);
#undef pNdotL
#undef clampMode
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
        NBL_CONST_REF_ARG(ray_dir_info_type) tangentSpaceL,
        const matrix3x3_type tangentFrame
    )
    {
        this_t retval;

        const vector3_type tsL = tangentSpaceL.getDirection();
        retval.L = tangentSpaceL.transform(tangentFrame);

        retval.TdotL = tsL.x;
        retval.BdotL = tsL.y;
        retval.NdotL = tsL.z;
        retval.NdotL2 = retval.NdotL*retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) L, const vector3_type N)
    {
        this_t retval;

        retval.L = L;
        retval.TdotL = bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN);
        retval.BdotL = bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN);
        retval.NdotL = nbl::hlsl::dot<vector3_type>(N,L.getDirection());
        retval.NdotL2 = retval.NdotL * retval.NdotL;

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) L, const vector3_type T, const vector3_type B, const vector3_type N)
    {
        this_t retval = create(L,N);

        retval.TdotL = nbl::hlsl::dot<vector3_type>(T,L.getDirection());
        retval.BdotL = nbl::hlsl::dot<vector3_type>(B,L.getDirection());

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

    ray_dir_info_type getL() NBL_CONST_MEMBER_FUNC { return L; }
    scalar_type getTdotL() NBL_CONST_MEMBER_FUNC { return TdotL; }
    scalar_type getTdotL2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotL(); return t*t; }
    scalar_type getBdotL() NBL_CONST_MEMBER_FUNC { return BdotL; }
    scalar_type getBdotL2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotL(); return t*t; }
    scalar_type getNdotL(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotL, _clamp);
    }
    scalar_type getNdotL2() NBL_CONST_MEMBER_FUNC { return NdotL2; }

    static this_t createInvalid()
    {
        this_t retval;
        retval.L.makeInvalid();
        return retval;
    }
    bool isValid() NBL_CONST_MEMBER_FUNC { return L.isValid(); }


    RayDirInfo L;

    scalar_type TdotL;
    scalar_type BdotL;
    scalar_type NdotL;
    scalar_type NdotL2;
};


#define NBL_CONCEPT_NAME ReadableIsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (eta, fresnel::OrientedEtas<vector<typename T::scalar_type,1> >)
NBL_CONCEPT_BEGIN(2)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define eta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getVdotL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getVdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getLdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getVdotHLdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getAbsNdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getNdotH2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.isTransmission()), ::nbl::hlsl::is_same_v, bool))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.isValid(eta)), ::nbl::hlsl::is_same_v, bool))
);
#undef eta
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME CreatableIsotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (iso, surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type >)
#define NBL_CONCEPT_PARAM_2 (pNdotV, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, SLightSample<ray_dir_info::SBasic<typename T::scalar_type> >)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (eta, fresnel::OrientedEtas<vector<typename T::scalar_type,1> >)
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
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(iso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,V,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(iso,_sample,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ReadableIsotropicMicrofacetCache, T))
);
#undef eta
#undef V
#undef _sample
#undef pNdotV
#undef iso
#undef cache
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T>)
struct SIsotropicMicrofacetCache
{
    using this_t = SIsotropicMicrofacetCache<T>;
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;
    using monochrome_type = vector<scalar_type, 1>;

    // always valid because its specialized for the reflective case
    static this_t createForReflection(const scalar_type NdotV, const scalar_type NdotL, const scalar_type VdotL, NBL_REF_ARG(scalar_type) LplusV_rcpLen)
    {
        scalar_type unoriented_LplusV_rcpLen = rsqrt<scalar_type>(2.0 + 2.0 * VdotL);

        this_t retval;
        retval.VdotL = VdotL;
        scalar_type NdotLplusVdotL = NdotL + NdotV;
        scalar_type oriented_LplusV_rcpLen = ieee754::flipSign<scalar_type>(unoriented_LplusV_rcpLen, NdotLplusVdotL < scalar_type(0.0)); 
        retval.VdotH = oriented_LplusV_rcpLen * VdotL + oriented_LplusV_rcpLen;
        retval.LdotH = retval.VdotH;
        retval.absNdotH = NdotLplusVdotL * oriented_LplusV_rcpLen;
        retval.NdotH2 = retval.absNdotH * retval.absNdotH;
        LplusV_rcpLen = unoriented_LplusV_rcpLen;

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
        return createForReflection(interaction.getNdotV(), _sample.getNdotL(), hlsl::dot<vector3_type>(interaction.getV().getDirection(), _sample.getL().getDirection()));
    }

    // transmissive cases need to be checked if the path is valid before usage
    static this_t create(const bool transmitted, NBL_CONST_REF_ARG(ComputeMicrofacetNormal<scalar_type>) computeMicrofacetNormal, const scalar_type VdotL,
        const vector3_type N, NBL_REF_ARG(vector3_type) H)
    {
        this_t retval;
        retval.VdotL = VdotL;
        H = computeMicrofacetNormal.normalized(transmitted);
        scalar_type NdotH = hlsl::dot<vector3_type>(N, H);
        H = ieee754::flipSign<vector3_type>(H, NdotH < scalar_type(0.0));
        retval.absNdotH = hlsl::abs(NdotH);

        // not coming from the medium (reflected) OR
        // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) )
        const bool valid = ComputeMicrofacetNormal<scalar_type>::isValidMicrofacet(transmitted, VdotL, retval.absNdotH, computeMicrofacetNormal.orientedEta);
        if (valid)
        {
            retval.VdotH = hlsl::dot<vector3_type>(computeMicrofacetNormal.V,H);
            retval.LdotH = hlsl::dot<vector3_type>(computeMicrofacetNormal.L,H);
            retval.NdotH2 = retval.absNdotH * retval.absNdotH;
        }
        else
            retval.absNdotH = bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN);
        return retval;
    }

    static this_t create(
        const vector3_type V, const vector3_type L, const vector3_type N,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas, NBL_REF_ARG(vector3_type) H)
    {
        this_t retval;
        const scalar_type NdotV = hlsl::dot<vector3_type>(N, V);
        const scalar_type NdotL = hlsl::dot<vector3_type>(N, L);
        const scalar_type VdotL = hlsl::dot<vector3_type>(V, L);
        const bool transmitted = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(NdotV,NdotL);

        ComputeMicrofacetNormal<scalar_type> computeMicrofacetNormal = ComputeMicrofacetNormal<scalar_type>::create(V,L,N,1.0);
        computeMicrofacetNormal.orientedEta = orientedEtas;
        
        return create(transmitted, computeMicrofacetNormal, VdotL, N, H);
    }

    static this_t create(
        const vector3_type V, const vector3_type L, const vector3_type N,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas)
    {
        vector3_type dummy;
        return create(V, L, N, orientedEtas, dummy);
    }

    template<class IsotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Isotropic<IsotropicInteraction> && LightSample<LS>)
    static this_t create(
        NBL_CONST_REF_ARG(IsotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas, NBL_REF_ARG(vector3_type) H)
    {
        const vector3_type V = interaction.getV().getDirection();
        const vector3_type L = _sample.getL().getDirection();
        const vector3_type N = interaction.getN();

        const bool transmitted = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(interaction.getNdotV(),_sample.getNdotL());

        ComputeMicrofacetNormal<scalar_type> computeMicrofacetNormal = ComputeMicrofacetNormal<scalar_type>::create(V,L,N,1.0);
        computeMicrofacetNormal.orientedEta = orientedEtas;
        
        return create(transmitted, computeMicrofacetNormal, hlsl::dot<vector3_type>(V, L), N, H);
    }

    template<class IsotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Isotropic<IsotropicInteraction> && LightSample<LS>)
    static this_t create(
        NBL_CONST_REF_ARG(IsotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas)
    {
        vector3_type dummy;
        return create(interaction,_sample,orientedEtas,dummy);
    }

    // note on usage: assert sign(VdotH)==sign(NdotV) and similar for L
    // problem is that BRDF GGX and friends can generate unmasked H, which is actually backfacing towards L (non VNDF variants can even generate VdotH<0)
    // similar with BSDF sampling, as fresnel can be high while reflection can be invalid, or low while refraction would be invalid too
    bool isTransmission() NBL_CONST_MEMBER_FUNC { return getVdotHLdotH() < scalar_type(0.0); }

    bool isValid(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas) NBL_CONST_MEMBER_FUNC
    {
        return ComputeMicrofacetNormal<scalar_type>::isValidMicrofacet(isTransmission(), VdotL, absNdotH, orientedEtas);
    }

    scalar_type getVdotL() NBL_CONST_MEMBER_FUNC { return VdotL; }
    scalar_type getVdotH() NBL_CONST_MEMBER_FUNC { return VdotH; }
    scalar_type getLdotH() NBL_CONST_MEMBER_FUNC { return LdotH; }
    scalar_type getVdotHLdotH() NBL_CONST_MEMBER_FUNC { return getVdotH() * getLdotH(); }
    scalar_type getAbsNdotH() NBL_CONST_MEMBER_FUNC { return absNdotH; }
    scalar_type getNdotH2() NBL_CONST_MEMBER_FUNC { return NdotH2; }

    scalar_type VdotL;
    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type absNdotH;
    scalar_type NdotH2;
};


#define NBL_CONCEPT_NAME AnisotropicMicrofacetCache
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (cache, T)
#define NBL_CONCEPT_PARAM_1 (aniso, surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type > >)
#define NBL_CONCEPT_PARAM_2 (pNdotL, typename T::scalar_type)
#define NBL_CONCEPT_PARAM_3 (_sample, SLightSample<ray_dir_info::SBasic<typename T::scalar_type> >)
#define NBL_CONCEPT_PARAM_4 (V, typename T::vector3_type)
#define NBL_CONCEPT_PARAM_5 (b0, bool)
#define NBL_CONCEPT_PARAM_6 (eta, fresnel::OrientedEtas<vector<typename T::scalar_type, 1> >)
#define NBL_CONCEPT_PARAM_7 (rcp_eta, fresnel::OrientedEtaRcps<vector<typename T::scalar_type, 1> >)
NBL_CONCEPT_BEGIN(8)
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define pNdotL NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define V NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define b0 NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define eta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
#define rcp_eta NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_7
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(ReadableIsotropicMicrofacetCache, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getTdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getTdotH2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getBdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((cache.getBdotH2()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(V,V)), ::nbl::hlsl::is_same_v, T))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,b0,rcp_eta)), ::nbl::hlsl::is_same_v, T))    // TODO: refuses to compile when arg4 is rcp_eta for some reason, eta is fine
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createForReflection(V,V,pNdotL)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template createForReflection<surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type > >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(aniso,_sample)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::create(V,V,V,V,V,eta,V)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::template create<surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<typename T::scalar_type>, typename T::vector3_type > >,SLightSample<ray_dir_info::SBasic<typename T::scalar_type> > >(aniso,_sample,eta)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((T::createPartial(pNdotL,pNdotL,pNdotL,b0,rcp_eta)), ::nbl::hlsl::is_same_v, T))
    ((NBL_CONCEPT_REQ_EXPR)(cache.fillTangents(V,V,V)))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(CreatableIsotropicMicrofacetCache, typename T::isocache_type))
);
#undef rcp_eta
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

    using ray_dir_info_type = ray_dir_info::SBasic<scalar_type>;
    using anisotropic_type = surface_interactions::SAnisotropic<ray_dir_info_type>;
    using isocache_type = SIsotropicMicrofacetCache<U>;
    using sample_type = SLightSample<ray_dir_info_type>;

    // always valid by construction
    static this_t createForReflection(const vector3_type tangentSpaceV, const vector3_type tangentSpaceH)
    {
        this_t retval;
        retval.iso_cache.VdotH = nbl::hlsl::dot<vector3_type>(tangentSpaceV,tangentSpaceH);
        retval.iso_cache.VdotL = scalar_type(2.0) * retval.iso_cache.VdotH * retval.iso_cache.VdotH - scalar_type(1.0);
        retval.iso_cache.LdotH = retval.iso_cache.getVdotH();
        assert(tangentSpaceH.z >= scalar_type(0.0));
        retval.iso_cache.absNdotH = tangentSpaceH.z;
        retval.iso_cache.NdotH2 = retval.iso_cache.getAbsNdotH() * retval.iso_cache.getAbsNdotH();
        retval.TdotH = tangentSpaceH.x;
        retval.BdotH = tangentSpaceH.y;

        return retval;
    }
    static this_t create(
        const vector3_type tangentSpaceV,
        const vector3_type tangentSpaceH,
        const bool transmitted,
        NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<monochrome_type>) rcpOrientedEta
    )
    {
        this_t retval = createForReflection(tangentSpaceV,tangentSpaceH);
        if (transmitted)
        {
            Refract<scalar_type> r = Refract<scalar_type>::create(tangentSpaceV, tangentSpaceH);
            retval.iso_cache.LdotH = r.getNdotT(rcpOrientedEta.value2[0]);
            retval.iso_cache.VdotL = retval.iso_cache.VdotH * (retval.iso_cache.LdotH - rcpOrientedEta.value[0] + retval.iso_cache.VdotH * rcpOrientedEta.value[0]);
        }

        return retval;
    }
    // always valid because its specialized for the reflective case
    static this_t createForReflection(const vector3_type tangentSpaceV, const vector3_type tangentSpaceL, const scalar_type VdotL)
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
        return createForReflection(interaction.getTangentSpaceV(), _sample.getTangentSpaceL(), hlsl::dot<vector3_type>(interaction.getV().getDirection(), _sample.getL().getDirection()));
    }
    // transmissive cases need to be checked if the path is valid before usage
    static this_t create(
        const vector3_type V, const vector3_type L,
        const vector3_type T, const vector3_type B, const vector3_type N,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas, NBL_REF_ARG(vector3_type) H
    )
    {
        isocache_type iso = (isocache_type)retval;
        const bool valid = isocache_type::compute(iso,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
        retval = (this_t)iso;
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(B,H);
        }
        return valid;
    }
    template<class AnisotropicInteraction, class LS NBL_FUNC_REQUIRES(surface_interactions::Anisotropic<AnisotropicInteraction> && LightSample<LS>)
    static this_t create(
        NBL_CONST_REF_ARG(AnisotropicInteraction) interaction,
        NBL_CONST_REF_ARG(LS) _sample,
        NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas
    )
    {
        vector3_type H;
        const bool valid = isocache_type::compute(iso,interaction,_sample,eta,H);
        retval = (this_t)iso;
        if (valid)
        {
            retval.TdotH = nbl::hlsl::dot<vector3_type>(interaction.T,H);
            retval.BdotH = nbl::hlsl::dot<vector3_type>(interaction.B,H);
        }
        return valid;
    }

    void fillTangents(const vector3_type T, const vector3_type B, const vector3_type H)
    {
        TdotH = hlsl::dot(T, H);
        BdotH = hlsl::dot(B, H);
    }

    scalar_type getVdotL() NBL_CONST_MEMBER_FUNC { return iso_cache.getVdotL(); }
    scalar_type getVdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.getVdotH(); }
    scalar_type getLdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.getLdotH(); }
    scalar_type getVdotHLdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.getVdotHLdotH(); }
    scalar_type getAbsNdotH() NBL_CONST_MEMBER_FUNC { return iso_cache.getAbsNdotH(); }
    scalar_type getNdotH2() NBL_CONST_MEMBER_FUNC { return iso_cache.getNdotH2(); }
    bool isTransmission() NBL_CONST_MEMBER_FUNC { return iso_cache.isTransmission(); }

    bool isValid(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEtas) NBL_CONST_MEMBER_FUNC
    {
        return iso_cache.isValid(orientedEtas);
    }

    scalar_type getTdotH() NBL_CONST_MEMBER_FUNC { return TdotH; }
    scalar_type getTdotH2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotH(); return t*t; }
    scalar_type getBdotH() NBL_CONST_MEMBER_FUNC { return BdotH; }
    scalar_type getBdotH2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotH(); return t*t; }

    isocache_type iso_cache;
    scalar_type TdotH;
    scalar_type BdotH;
};


namespace bxdf_concepts
{
namespace impl
{

#define NBL_CONCEPT_NAME bxdf_common_typdefs
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
NBL_CONCEPT_BEGIN(1)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
);
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME bxdf_common
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_2 (aniso, typename T::anisotropic_interaction_type)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(bxdf_common_typdefs, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(_sample, aniso)), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.pdf(_sample, aniso)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(_sample, aniso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(LightSample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::FloatingPointLikeVectorial, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_interaction_type))
);
#undef aniso
#undef _sample
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME iso_bxdf_common
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_2 (iso, typename T::isotropic_interaction_type)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(bxdf_common, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(_sample, iso)), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.pdf(_sample, iso)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(_sample, iso)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::FloatingPointLikeVectorial, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_interaction_type))
);
#undef iso
#undef _sample
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

#define NBL_CONCEPT_NAME BRDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 2>)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef u
#undef aniso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME BSDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 3>)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef u
#undef aniso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME IsotropicBRDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 2>)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::iso_bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(iso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef u
#undef iso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME IsotropicBSDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 3>)
NBL_CONCEPT_BEGIN(3)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::iso_bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(iso,u)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef u
#undef iso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T>
NBL_BOOL_CONCEPT BxDF = BRDF<T> || BSDF<T>;
template<typename T>
NBL_BOOL_CONCEPT IsotropicBxDF = IsotropicBRDF<T> || IsotropicBSDF<T>;


namespace impl
{
#define NBL_CONCEPT_NAME microfacet_bxdf_common
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_2 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_3 (anisocache, typename T::anisocache_type)
NBL_CONCEPT_BEGIN(4)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define anisocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(bxdf_common_typdefs, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(_sample, aniso, anisocache)), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.pdf(_sample, aniso, anisocache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(_sample, aniso, anisocache)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(LightSample, typename T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::FloatingPointLikeVectorial, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Anisotropic, typename T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(AnisotropicMicrofacetCache, typename T::anisocache_type))
);
#undef anisocache
#undef aniso
#undef _sample
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME iso_microfacet_bxdf_common
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (_sample, typename T::sample_type)
#define NBL_CONCEPT_PARAM_2 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_3 (isocache, typename T::isocache_type)
NBL_CONCEPT_BEGIN(4)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define isocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(microfacet_bxdf_common, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.eval(_sample, iso, isocache)), ::nbl::hlsl::is_same_v, typename T::spectral_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.pdf(_sample, iso, isocache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.quotient_and_pdf(_sample, iso, isocache)), ::nbl::hlsl::is_same_v, typename T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(surface_interactions::Isotropic, typename T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(CreatableIsotropicMicrofacetCache, typename T::isocache_type))
);
#undef isocache
#undef iso
#undef _sample
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

// unified param struct for calls to BxDF::eval, BxDF::pdf, BxDF::quotient_and_pdf
template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBxDFParams
{
    using this_t = SBxDFParams<Scalar>;

#define NBL_CONCEPT_NAME MicrofacetBSDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (aniso, typename T::anisotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 3>)
#define NBL_CONCEPT_PARAM_3 (anisocache, typename T::anisocache_type)
NBL_CONCEPT_BEGIN(4)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define aniso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define anisocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::microfacet_bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(aniso,u,anisocache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef anisocache
#undef u
#undef aniso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    static this_t create(LightSample _sample, Aniso interaction, BxDFClampMode clamp = BCM_NONE)
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

#define NBL_CONCEPT_NAME IsotropicMicrofacetBSDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (bxdf, T)
#define NBL_CONCEPT_PARAM_1 (iso, typename T::isotropic_interaction_type)
#define NBL_CONCEPT_PARAM_2 (u, vector<typename T::scalar_type, 3>)
#define NBL_CONCEPT_PARAM_3 (isocache, typename T::isocache_type)
NBL_CONCEPT_BEGIN(4)
#define bxdf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define iso NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define u NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define isocache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(impl::iso_microfacet_bxdf_common, T))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((bxdf.generate(iso,u,isocache)), ::nbl::hlsl::is_same_v, typename T::sample_type))
);
#undef isocache
#undef u
#undef iso
#undef bxdf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

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
    static this_t create(LightSample _sample, Aniso interaction, Cache cache, BxDFClampMode clamp = BCM_NONE)
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

// fresnel stuff
namespace impl
{
template<typename T>
NBL_BOOL_CONCEPT MicrofacetBxDF = MicrofacetBRDF<T> || MicrofacetBSDF<T>;
template<typename T>
NBL_BOOL_CONCEPT IsotropicMicrofacetBxDF = IsotropicMicrofacetBRDF<T> || IsotropicMicrofacetBSDF<T>;
}


template<typename T, typename P=T>
using quotient_and_pdf_scalar = sampling::quotient_and_pdf<vector<T, 1>, P>;
template<typename T, typename P=T>
using quotient_and_pdf_rgb = sampling::quotient_and_pdf<vector<T, 3>, P>;

namespace impl
{
template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T> && sizeof(T)<8)
struct beta
{
    // beta function specialized for Cook Torrance BTDFs
    // uses modified Stirling's approximation for log2 up to 6th polynomial
    static T __series_part(T x)
    {
        const T r = T(1.0) / x;
        const T r2 = r * r;
        T ser = T(0.0);
        if (sizeof(T) > 2)
        {
            ser = -T(691.0/360360.0) * r2 + T(5.0/5940.0);
            ser = ser * r2 - T(1.0/1680.0);
            ser = ser * r2 + T(1.0/1260.0);
        }
        ser = ser * r2 - T(1.0/360.0);
        ser = ser * r2 + T(1.0/12.0);
        return ser * r - x;
    }

    // removed values that cancel out in beta
    static T __call_wo_check(T x, T y)
    {
        const T l2x = hlsl::log2<T>(x);
        const T l2y = hlsl::log2<T>(y);
        const T l2xy = hlsl::log2<T>(x+y);

        return hlsl::exp2<T>((x - T(0.5)) * l2x + (y - T(0.5)) * l2y - (x + y - T(0.5)) * l2xy +
            numbers::inv_ln2<T> * (__series_part(x) + __series_part(y) - __series_part(x+y)) + T(1.32574806473616));
    }

    static T __call(T x, T y)
    {
        assert(x >= T(0.999) && y >= T(0.999));

// currently throws a boost preprocess error, see: https://github.com/Devsh-Graphics-Programming/Nabla/issues/932
// #ifdef __HLSL_VERSION
        #pragma dxc diagnostic push
		#pragma dxc diagnostic ignored "-Wliteral-range"
// #endif
		const T thresholds[4] = { 0, 5e5, 1e6, 1e15 };	// threshold values gotten from testing when the function returns nan/inf/1
// #ifdef __HLSL_VERSION
        #pragma dxc diagnostic pop
// #endif
		if (x+y > thresholds[mpl::find_lsb_v<sizeof(T)>])
			return T(0.0);

        return __call_wo_check(x, y);
    }
};
}

template<typename T>
T beta(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return impl::beta<T>::__call(x, y)/impl::beta<T>::__call(1.0, 1.0);
}

template<typename T>
T beta_wo_check(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return impl::beta<T>::__call_wo_check(x, y)/impl::beta<T>::__call_wo_check(1.0, 1.0);
}

}
}
}

#endif
