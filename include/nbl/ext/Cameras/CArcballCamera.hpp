// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_ARCBALL_CAMERA_HPP_
#define _C_ARCBALL_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Target-relative camera with planar target translation and bounded arcball orbiting.
///
/// The runtime state is inherited from `CSphericalTargetCamera`. Translation
/// moves the target in the current view plane. Rotation updates orbit yaw and
/// pitch under a symmetric pitch limit.
class CArcballCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CArcballCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv.y = std::clamp(m_orbitUv.y, MinPitch, MaxPitch);
        applyPose();
    }
    ~CArcballCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Apply one frame of semantic translation and rotation input to the arcball rig.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        if (referenceFrame)
        {
            CReferenceTransform reference = {};
            SCameraTargetRelativeState resolvedState = {};
            if (!tryExtractReferenceTransform(reference, referenceFrame) ||
                !tryResolveReferenceTargetRelativeState(reference, resolvedState))
            {
                return false;
            }

            resolvedState.orbitUv.y = std::clamp(resolvedState.orbitUv.y, MinPitch, MaxPitch);
            adoptTargetRelativeState(resolvedState);
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv.x += deltaRotation.y;
        m_orbitUv.y = std::clamp(m_orbitUv.y + deltaRotation.x, MinPitch, MaxPitch);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        const auto basis = computeBasis(m_orbitUv, m_distance);
        applyPlanarTargetTranslation(deltaTranslation, basis);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Arcball; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Arcball Camera"; }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

private:

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
