// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_TURNTABLE_CAMERA_HPP_
#define _C_TURNTABLE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Target-relative camera that behaves like a classic turntable around a fixed target.
///
/// The camera exposes yaw, bounded pitch, and distance changes while keeping the
/// target fixed in space and avoiding arbitrary planar target translation.
class CTurntableCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CTurntableCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv.y = std::clamp(m_orbitUv.y, MinPitch, MaxPitch);
        applyPose();
    }
    ~CTurntableCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Apply one frame of yaw, bounded pitch, and distance input around the tracked target.
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

        const double deltaYaw = scaleVirtualRotation(impulse.dVirtualRotation.y);
        const double deltaPitch = scaleVirtualRotation(impulse.dVirtualRotation.x);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv.x += deltaYaw;
        m_orbitUv.y = std::clamp(m_orbitUv.y + deltaPitch, MinPitch, MaxPitch);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Turntable; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Turntable Camera"; }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

private:

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::TurntablePitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
