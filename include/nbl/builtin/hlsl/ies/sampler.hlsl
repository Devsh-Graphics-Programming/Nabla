// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_
#define _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/algorithm.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"
#include "nbl/builtin/hlsl/math/octahedral.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/ies/profile.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ies
{
namespace concepts
{
#define NBL_CONCEPT_NAME IESAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (accessor_t)
#define NBL_CONCEPT_PARAM_0 (accessor, accessor_t)
NBL_CONCEPT_BEGIN(1)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define req_key_t uint32_t
#define req_key_t2 uint32_t2
#define req_angle_t float32_t
#define req_candela_t float32_t
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(accessor_t::angle_t))
    ((NBL_CONCEPT_REQ_TYPE)(accessor_t::candela_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((req_angle_t(0)), is_same_v, typename accessor_t::angle_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((req_candela_t(0)), is_same_v, typename accessor_t::candela_t))

    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.getProperties()), is_same_v, ProfileProperties))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.value(req_key_t2(0, 0))), is_same_v, typename accessor_t::candela_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.vAnglesCount()), is_same_v, req_key_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.hAnglesCount()), is_same_v, req_key_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.vAngle(req_key_t(0))), is_same_v, typename accessor_t::angle_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.hAngle(req_key_t(0))), is_same_v, typename accessor_t::angle_t))
);
#undef accessor
#undef req_key_t
#undef req_key_t2
#undef req_angle_t
#undef req_candela_t
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename accessor_t>
NBL_BOOL_CONCEPT IsIESAccessor = IESAccessor<accessor_t>;
}

template<typename Accessor NBL_FUNC_REQUIRES(concepts::IsIESAccessor<Accessor>)
struct CandelaSampler
{
    using accessor_t = Accessor;
    using angle_t = typename accessor_t::angle_t;
    using candela_t = typename accessor_t::candela_t;
    using symmetry_t = ProfileProperties::LuminairePlanesSymmetry;
    using polar_t = math::Polar<float32_t>;
    using octahedral_t = math::OctahedralTransform<float32_t>;
    using vector2_type = float32_t2;

    vector2_type halfMinusHalfPixel;

    static inline CandelaSampler create(NBL_CONST_REF_ARG(vector2_type) lastTexelRcp)
    {
        CandelaSampler retval;
        retval.halfMinusHalfPixel = vector2_type(0.5f, 0.5f) / (vector2_type(1.f, 1.f) + lastTexelRcp);
        return retval;
    }

    inline candela_t operator()(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(polar_t) polar) NBL_CONST_MEMBER_FUNC
    {
        assert(polar.theta >= float32_t(0.0) && polar.theta <= numbers::pi<float32_t>);
        assert(hlsl::abs(polar.phi) <= numbers::pi<float32_t> * float32_t(2.0));

        const symmetry_t symmetry = accessor.getProperties().getSymmetry();
        const angle_t vAngle = degrees(polar.theta);
        const angle_t hAngle = degrees(__wrapPhi(polar.phi, symmetry));

#define NBL_IES_DEF_ANGLE_ACC(T, EXPR) struct T { using value_type = angle_t; accessor_t acc; value_type operator[](uint32_t idx) NBL_CONST_MEMBER_FUNC { return EXPR; } };

        NBL_IES_DEF_ANGLE_ACC(VAcc, acc.vAngle(idx))
        NBL_IES_DEF_ANGLE_ACC(HAcc, acc.hAngle(idx))

        VAcc vAcc; vAcc.acc = accessor; HAcc hAcc; hAcc.acc = accessor;

#undef NBL_IES_DEF_ANGLE_ACC

        const uint32_t vCount = accessor.vAnglesCount();
        const uint32_t hCount = accessor.hAnglesCount();
        const angle_t vABack = vAcc[vCount - 1u];
        if (vAngle > vABack)
            return candela_t(0);

        const uint32_t vUbRaw = __upperBound(vAcc, vCount, vAngle);
        const uint32_t vLb = __lowerFromUpper(vUbRaw);
        const uint32_t vUb = __clampUpper(vUbRaw, vCount);

        const bool isotropic = (symmetry == symmetry_t::ISOTROPIC);
        const uint32_t hUbRaw = isotropic ? 0u : __upperBound(hAcc, hCount, hAngle);
        const uint32_t hLb = isotropic ? 0u : __lowerFromUpper(hUbRaw);
        const uint32_t hUb = isotropic ? 0u : __clampUpper(hUbRaw, hCount);

        const angle_t uReciprocal = (hUb == hLb) ? angle_t(1) : angle_t(1) / (hAcc[hUb] - hAcc[hLb]);
        const angle_t vReciprocal = (vUb == vLb) ? angle_t(1) : angle_t(1) / (vAcc[vUb] - vAcc[vLb]);

        const angle_t u = (hAngle - hAcc[hLb]) * uReciprocal;
        const angle_t v = (vAngle - vAcc[vLb]) * vReciprocal;

        const candela_t s0 = accessor.value(uint32_t2(hLb, vLb)) * (angle_t(1) - v) + accessor.value(uint32_t2(hLb, vUb)) * v;
        const candela_t s1 = accessor.value(uint32_t2(hUb, vLb)) * (angle_t(1) - v) + accessor.value(uint32_t2(hUb, vUb)) * v;

        return s0 * (angle_t(1) - u) + s1 * u;
    }

    inline candela_t operator()(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(float32_t2) uv) NBL_CONST_MEMBER_FUNC
    {
        const float32_t3 dir = octahedral_t::uvToDir(uv, halfMinusHalfPixel);
        const polar_t polar = polar_t::createFromCartesian(dir);
        return operator()(accessor, polar);
    }

    template<typename View>
    static inline uint32_t __upperBound(NBL_REF_ARG(View) view, const uint32_t count, const angle_t angle) { return nbl::hlsl::upper_bound(view, 0u, count, angle); }

    static inline uint32_t __lowerFromUpper(const uint32_t ubRaw) { return ubRaw > 0u ? (ubRaw - 1u) : 0u; }

    static inline uint32_t __clampUpper(const uint32_t ubRaw, const uint32_t count) { return ubRaw < count ? ubRaw : (count - 1u); }

    static inline angle_t __wrapPhi(const angle_t phi, const symmetry_t symmetry)
    {
        switch (symmetry)
        {
            case symmetry_t::ISOTROPIC: //! axial symmetry
                return angle_t(0.0);
            case symmetry_t::QUAD_SYMETRIC: //! phi MIRROR_REPEAT wrap onto [0, 90] degrees range
            {
                const angle_t HalfPI = numbers::pi<angle_t> * angle_t(0.5);
                angle_t wrapPhi = hlsl::abs(phi); //! first MIRROR
                if (wrapPhi > HalfPI) //! then REPEAT
                    wrapPhi = hlsl::clamp(HalfPI - (wrapPhi - HalfPI), angle_t(0), HalfPI);
                return wrapPhi; //! eg. maps (in degrees) 91,269,271 -> 89 and 179,181,359 -> 1
            }
            case symmetry_t::HALF_SYMETRIC: //! phi MIRROR wrap onto [0, 180] degrees range
            case symmetry_t::OTHER_HALF_SYMMETRIC: //! eg. maps (in degress) 181 -> 179 or 359 -> 1
                return hlsl::abs(phi);
            case symmetry_t::NO_LATERAL_SYMMET: //! plot onto whole (in degress) [0, 360] range
            {
                const angle_t TwicePI = numbers::pi<angle_t> * angle_t(2.0);
                return (phi < angle_t(0)) ? (phi + TwicePI) : phi;
            }
        }

        return bit_cast<angle_t>(numeric_limits<float32_t>::quiet_NaN);
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_
