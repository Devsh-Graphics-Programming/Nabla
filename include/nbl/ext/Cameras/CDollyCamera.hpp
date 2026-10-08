// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_DOLLY_CAMERA_HPP_
#define _C_DOLLY_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera that translates the target in the full local camera basis.
///
/// Controls: `translate` moves the target through the camera's own frame (x right, y up, z forward), world
/// units, so the camera follows at its distance; `rotate.y` azimuth and `rotate.x` elevation of the camera
/// around the target, radians, the elevation clamped to the dolly pitch limit.
/// TODO: `Distance` is not accepted; the rig cannot change its distance from input, as before, until its intent
/// is decided.
class CDollyCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.angles.y = std::clamp(m_orbit.angles.y, MinPitch, MaxPitch);
        updateGimbal();
    }
    ~CDollyCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Orbit around the target through the position of `pose`, under the dolly pitch limit.
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
    virtual CameraKind getKind() const override { return CameraKind::Dolly; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Dolly Camera"; }

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::Translate | ECameraControlAxis::RotateX | ECameraControlAxis::RotateY;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        // the dolly moves the target through the full committed basis, forward component included
        // TODO: like the planar pan in the base rig, this delta is a fixed world-space length and should scale with distance
        const auto basis = m_gimbal.getBasis();
        const auto delta = CCameraMathUtilities::transformLocalVectorToWorldBasis(controls.translate, basis.right, basis.up, basis.forward);

        m_orbit.target += delta;
        m_orbit.angles.x += controls.rotate.y;
        m_orbit.angles.y = std::clamp(m_orbit.angles.y + controls.rotate.x, MinPitch, MaxPitch);

        return updateGimbal();
    }

public:
    /// @brief Pitch limit in radians. It stops 5 deg short of straight up and down.
    static inline constexpr double MaxPitch = 85.0 * (hlsl::numbers::pi<double> / 180.0);
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
