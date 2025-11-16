// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_
#define _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"
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
#define req_value_t float32_t
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(accessor_t::key_t))
    ((NBL_CONCEPT_REQ_TYPE)(accessor_t::key_t2))
    ((NBL_CONCEPT_REQ_TYPE)(accessor_t::value_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((req_key_t(0)), is_same_v, typename accessor_t::key_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((req_key_t2(0, 0)), is_same_v, typename accessor_t::key_t2))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((req_value_t(0)), is_same_v, typename accessor_t::value_t))

    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.vAnglesCount()), is_same_v, req_key_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.hAnglesCount()), is_same_v, req_key_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.symmetry()), is_same_v, ProfileProperties::LuminairePlanesSymmetry))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template vAngle<req_key_t>((req_key_t)0)), is_same_v, req_value_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template hAngle<req_key_t>((req_key_t)0)), is_same_v, req_value_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template value<req_key_t2>((req_key_t2)0)), is_same_v, req_value_t))
);
#undef accessor
#undef req_key_t
#undef req_key_t2
#undef req_value_t
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename accessor_t>
NBL_BOOL_CONCEPT IsIESAccessor = IESAccessor<accessor_t>;
}

template<typename Accessor NBL_FUNC_REQUIRES(concepts::IsIESAccessor<Accessor>)
struct CandelaSampler
{
    using accessor_t = Accessor;
    using value_t = typename accessor_t::value_t;
    using symmetry_t = ProfileProperties::LuminairePlanesSymmetry;

    static value_t sample(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(math::Polar<float32_t>) polar)
    {
        const symmetry_t symmetry = accessor.symmetry();
        const float32_t vAngle = degrees(polar.theta);
        const float32_t hAngle = degrees(wrapPhi(polar.phi, symmetry));

        const float32_t vABack = accessor.vAngle(accessor.vAnglesCount() - 1u);
        if (vAngle > vABack)
            return 0.f;

        const uint32_t j0 = getVLB(accessor, vAngle);
        const uint32_t j1 = getVUB(accessor, vAngle);
        const uint32_t i0 = (symmetry == symmetry_t::ISOTROPIC) ? 0u : getHLB(accessor, hAngle);
        const uint32_t i1 = (symmetry == symmetry_t::ISOTROPIC) ? 0u : getHUB(accessor, hAngle);

        const float32_t uReciprocal = ((i1 == i0) ? 1.f : 1.f / (accessor.hAngle(i1) - accessor.hAngle(i0)));
        const float32_t vReciprocal = ((j1 == j0) ? 1.f : 1.f / (accessor.vAngle(j1) - accessor.vAngle(j0)));

        const float32_t u = ((hAngle - accessor.hAngle(i0)) * uReciprocal);
        const float32_t v = ((vAngle - accessor.vAngle(j0)) * vReciprocal);

        const float32_t s0 = (accessor.value(uint32_t2(i0, j0)) * (1.f - v) + accessor.value(uint32_t2(i0, j1)) * v);
        const float32_t s1 = (accessor.value(uint32_t2(i1, j0)) * (1.f - v) + accessor.value(uint32_t2(i1, j1)) * v);

        return s0 * (1.f - u) + s1 * u;
    }

    static float32_t wrapPhi(const float32_t phi, const symmetry_t symmetry)
    {
        switch (symmetry)
        {
        case symmetry_t::ISOTROPIC: //! axial symmetry
            return 0.0f;
        case symmetry_t::QUAD_SYMETRIC: //! phi MIRROR_REPEAT wrap onto [0, 90] degrees range
        {
            NBL_CONSTEXPR float32_t M_HALF_PI = numbers::pi<float32_t> * 0.5f;
            float32_t wrapPhi = abs(phi); //! first MIRROR
            if (wrapPhi > M_HALF_PI) //! then REPEAT
                wrapPhi = hlsl::clamp(M_HALF_PI - (wrapPhi - M_HALF_PI), 0.f, M_HALF_PI);
            return wrapPhi; //! eg. maps (in degrees) 91,269,271 -> 89 and 179,181,359 -> 1
        }
        case symmetry_t::HALF_SYMETRIC: //! phi MIRROR wrap onto [0, 180] degrees range
        case symmetry_t::OTHER_HALF_SYMMETRIC: //! eg. maps (in degress) 181 -> 179 or 359 -> 1
            return abs(phi);
        case symmetry_t::NO_LATERAL_SYMMET: //! plot onto whole (in degress) [0, 360] range
        {
            NBL_CONSTEXPR float32_t M_TWICE_PI = numbers::pi<float32_t> *2.f;
            return (phi < 0.f) ? (phi + M_TWICE_PI) : phi;
        }
        }
        return 69.f;
    }

    struct impl_t
    {
        static uint32_t getVUB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
        {
            for (uint32_t i = 0u; i < accessor.vAnglesCount(); ++i)
                if (accessor.vAngle(i) > angle)
                    return i;
            return accessor.vAnglesCount();
        }

        static uint32_t getHUB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
        {
            for (uint32_t i = 0u; i < accessor.hAnglesCount(); ++i)
                if (accessor.hAngle(i) > angle)
                    return i;
            return accessor.hAnglesCount();
        }
    };

    static uint32_t getVLB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
    {
        return (uint32_t)hlsl::max((int64_t)impl_t::getVUB(accessor, angle) - 1ll, 0ll);
    }

    static uint32_t getHLB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
    {
        return (uint32_t)hlsl::max((int64_t)impl_t::getHUB(accessor, angle) - 1ll, 0ll);
    }

    static uint32_t getVUB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
    {
        return (uint32_t)hlsl::min((int64_t)impl_t::getVUB(accessor, angle), (int64_t)(accessor.vAnglesCount() - 1u));
    }

    static uint32_t getHUB(NBL_CONST_REF_ARG(accessor_t) accessor, const float32_t angle)
    {
        return (uint32_t)hlsl::min((int64_t)impl_t::getHUB(accessor, angle), (int64_t)(accessor.hAnglesCount() - 1u));
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_SAMPLER_INCLUDED_