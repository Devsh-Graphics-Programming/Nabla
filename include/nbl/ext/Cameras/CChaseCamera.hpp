// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CHASE_CAMERA_HPP_
#define _C_CHASE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera with planar target translation on the ground plane.
///
/// Controls: `translate.x` and `translate.z` move the target along the camera's right and forward flattened
/// onto the frame's horizon plane, world units, so the camera follows at its distance; `distance` along the
/// camera-target line, world units, clamped to the distance limits; `rotate.y` azimuth and `rotate.x`
/// elevation of the camera around the target, radians, the elevation clamped to the chase pitch envelope.
class CChaseCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CChaseCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, const SWorldFrame& frame = SWorldFrame::yUpLeftHanded())
        : base_t(position, target, frame)
    {
        m_orbit.angles.y = std::clamp(m_orbit.angles.y, MinPitch, MaxPitch);
        updateGimbal();
    }
    ~CChaseCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the target through the position of `pose`, under the chase pitch envelope.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        STargetOrbit orbit = {};
        if (!CCameraMathUtilities::tryBuildOrbitFromPosition(m_orbit.target, pose.position, MinDistance, MaxDistance, getFrame(), orbit))
            return false;

        orbit.angles.y = std::clamp(orbit.angles.y, MinPitch, MaxPitch);
        m_orbit = orbit;
        updateGimbal();
        return true;
    }

    virtual uint32_t getAcceptedControls() const override { return AcceptedControls; }
    virtual CameraKind getKind() const override { return CameraKind::Chase; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Chase Camera"; }

    static inline constexpr uint32_t AcceptedControls =
        ECameraControlAxis::TranslateX | ECameraControlAxis::TranslateZ | ECameraControlAxis::Distance |
        ECameraControlAxis::RotateX | ECameraControlAxis::RotateY;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        // chase translation stays in the horizon plane, so the committed basis loses its component along the frame's up
        // TODO: like the planar pan in the base rig, this delta is a fixed world-space length and should scale with distance
        const auto basis = getBasis();
        const auto& frame = getFrame();

        const auto planarForward = CCameraMathUtilities::safeNormalizeVec3(
            basis.forward - frame.up * hlsl::dot(basis.forward, frame.up),
            frame.forward);
        const auto planarRight = CCameraMathUtilities::safeNormalizeVec3(
            basis.right - frame.up * hlsl::dot(basis.right, frame.up),
            frame.right);

        m_orbit.target += planarRight * controls.translate.x + planarForward * controls.translate.z;
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);

        m_orbit.angles.x += controls.rotate.y;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + controls.rotate.x, MinPitch, MaxPitch);

        return updateGimbal();
    }

public:
    /// @brief Pitch range in radians, narrower than the other rigs so the followed subject stays readable.
    static inline constexpr double MaxPitch = 70.0 * (hlsl::numbers::pi<double> / 180.0);
    static inline constexpr double MinPitch = -60.0 * (hlsl::numbers::pi<double> / 180.0);
};

}

#endif
