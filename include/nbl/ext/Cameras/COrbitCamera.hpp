// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include <algorithm>
#include <cmath>
#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera whose state is one `STargetOrbit`.
///
/// Controls: `rotate.y` azimuth and `rotate.x` elevation of the camera around the target, radians, neither
/// clamped; `distance` along the camera-target line, world units, clamped to the distance limits. The target
/// stays where it is.
class COrbitCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    COrbitCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.distance = std::clamp(hlsl::length(m_orbit.target - position), MinDistance, MaxDistance);
        updateGimbal();
    }
    ~COrbitCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the current target through the position of `pose`.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, orbit))
            return false;

        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    virtual uint32_t getAcceptedControls() const override
    {
        return AcceptedControls;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Orbit;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Orbit Camera";
    }

    static inline constexpr hlsl::float64_t MinDistance = base_t::MinDistance;
    static inline constexpr hlsl::float64_t MaxDistance = base_t::MaxDistance;

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::RotateX | ECameraControlAxis::RotateY | ECameraControlAxis::Distance;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        m_orbit.angles += hlsl::float64_t2(controls.rotate.y, controls.rotate.x);
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);

        return updateGimbal();
    }
};

}

#endif // _C_ORBIT_CAMERA_HPP_
