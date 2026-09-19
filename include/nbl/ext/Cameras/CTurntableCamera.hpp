// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
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
/// Controls: `rotate.y` azimuth and `rotate.x` elevation of the camera around the target, radians, the
/// elevation clamped to the turntable pitch limit; `distance` along the camera-target line, world units,
/// clamped to the distance limits. The target stays where it is.
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

    virtual uint32_t getAcceptedControls() const override { return AcceptedControls; }
    virtual CameraKind getKind() const override { return CameraKind::Turntable; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Turntable Camera"; }

    static inline constexpr hlsl::float64_t MinDistance = base_t::MinDistance;
    static inline constexpr hlsl::float64_t MaxDistance = base_t::MaxDistance;

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::RotateX | ECameraControlAxis::RotateY | ECameraControlAxis::Distance;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        m_orbit.angles.x += controls.rotate.y;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + controls.rotate.x, MinPitch, MaxPitch);
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);

        return updateGimbal();
    }

private:
    static inline constexpr double MaxPitch = SCameraViewRigDefaults::TurntablePitchLimitRad;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
