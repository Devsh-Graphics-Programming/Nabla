// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_ARCBALL_CAMERA_HPP_
#define _C_ARCBALL_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
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
        m_orbit.angles.y = std::clamp(m_orbit.angles.y, MinPitch, MaxPitch);
        updateGimbal();
    }
    ~CArcballCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the target through the position of `pose`, under the arcball pitch limit.
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

    /// @brief Apply one frame of semantic translation and rotation input to the arcball rig.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const auto deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbit.angles.x += deltaRotation.y;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + deltaRotation.x, MinPitch, MaxPitch);
        m_orbit.distance = std::clamp(m_orbit.distance + deltaDistance, MinDistance, MaxDistance);
        applyPlanarTargetTranslation(deltaTranslation);

        return updateGimbal();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Arcball; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Arcball Camera"; }

    static inline constexpr hlsl::float64_t MinDistance = base_t::MinDistance;
    static inline constexpr hlsl::float64_t MaxDistance = base_t::MaxDistance;

private:

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
