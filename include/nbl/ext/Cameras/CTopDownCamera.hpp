// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_TOPDOWN_CAMERA_HPP_
#define _C_TOPDOWN_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Target-relative camera constrained to look straight down at the tracked target.
///
/// Controls: `rotate.y` azimuth of the camera around the target, radians, unclamped; `distance` along the
/// camera-target line, world units, clamped to the distance limits; `translate.x` and `translate.y` move the
/// target in the current view plane (x right, y up), world units, so the camera slides with it. The pitch is
/// held at the top-down angle.
class CTopDownCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CTopDownCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbit.angles.y = TopDownPitch;
        updateGimbal();
    }
    ~CTopDownCamera() = default;

    const CCameraGimbal& getGimbal() override { return m_gimbal; }

    using base_t::setPose;

    /// @brief Take the distance of `pose` and the yaw its orientation encodes, holding the pitch at the top-down angle.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        const auto distance = hlsl::length(pose.position - m_orbit.target);
        if (!CCameraMathUtilities::isFiniteScalar(distance) ||
            distance <= static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon))
        {
            return false;
        }

        m_orbit.distance = std::clamp(distance, MinDistance, MaxDistance);
        m_orbit.angles.x = resolveTopDownYaw(pose.orientation, m_orbit.angles.x);
        m_orbit.angles.y = TopDownPitch;
        updateGimbal();
        return true;
    }

    virtual uint32_t getAcceptedControls() const override { return AcceptedControls; }
    virtual CameraKind getKind() const override { return CameraKind::TopDown; }
    /// @brief Return the stable user-facing identifier for this concrete camera kind.
    virtual std::string_view getIdentifier() const override { return "Top-Down Camera"; }

    static inline constexpr uint32_t AcceptedControls =
        ECameraControlAxis::RotateY | ECameraControlAxis::Distance | ECameraControlAxis::TranslateX | ECameraControlAxis::TranslateY;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        m_orbit.angles.x += controls.rotate.y;
        m_orbit.angles.y = TopDownPitch;
        m_orbit.distance = std::clamp(m_orbit.distance + controls.distance, MinDistance, MaxDistance);
        applyPlanarTargetTranslation(controls.translate);

        return updateGimbal();
    }

private:
    /// @brief Recover the yaw a top-down orientation encodes, falling back when the pose carries none.
    ///
    /// TODO: this derivation assumes an elevation of +90 deg, while `TopDownPitch` is -90 (see the TODO on
    /// `SCameraViewRigDefaults::TopDownPitchDeg`); fixing that sign fixes the 180 deg error here too.
    static inline double resolveTopDownYaw(const hlsl::math::quaternion<hlsl::float64_t>& orientation, const double fallbackYaw)
    {
        const auto basis = CCameraMathUtilities::getOrientationBasis(orientation);
        // looking straight down, the camera up vector lies in the ground plane; with +Y up that is the XZ plane,
        // where `makeSphericalUpFromOrbit` gives up = (-sin(yaw), 0, -cos(yaw))
        const auto planarUp = hlsl::float64_t2(basis.up.x, basis.up.z);
        constexpr auto Epsilon = static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon);
        if (!CCameraMathUtilities::isNearlyZeroVector(planarUp, Epsilon))
            return hlsl::atan2(-planarUp.x, -planarUp.y);

        // the same pose gives right = (-cos(yaw), 0, sin(yaw))
        const auto planarRight = hlsl::float64_t2(basis.right.x, basis.right.z);
        if (!CCameraMathUtilities::isNearlyZeroVector(planarRight, Epsilon))
            return hlsl::atan2(planarRight.y, -planarRight.x);

        return fallbackYaw;
    }

    static inline constexpr double TopDownPitch = SCameraViewRigDefaults::TopDownPitchRad;
};

}

#endif
