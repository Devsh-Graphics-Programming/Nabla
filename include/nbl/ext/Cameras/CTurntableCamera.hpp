// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_TURNTABLE_CAMERA_HPP_
#define _C_TURNTABLE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
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
        m_orbit.angles.y = std::clamp(m_orbit.angles.y, MinPitch, MaxPitch);
        updateGimbal();
    }
    ~CTurntableCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the target through the position of `pose`, under the turntable pitch limit.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, orbit))
            return false;

        orbit.angles.y = std::clamp(orbit.angles.y, MinPitch, MaxPitch);
        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    /// @brief Apply one frame of yaw, bounded pitch, and distance input around the tracked target.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);

        const auto deltaYaw = scaleVirtualRotation(impulse.dVirtualRotation.y);
        const auto deltaPitch = scaleVirtualRotation(impulse.dVirtualRotation.x);
        const auto deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbit.angles.x += deltaYaw;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + deltaPitch, MinPitch, MaxPitch);
        m_orbit.distance = std::clamp(m_orbit.distance + deltaDistance, MinDistance, MaxDistance);

        return updateGimbal();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Turntable; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Turntable Camera"; }

    static inline constexpr hlsl::float64_t MinDistance = base_t::MinDistance;
    static inline constexpr hlsl::float64_t MaxDistance = base_t::MaxDistance;

private:

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::TurntablePitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
