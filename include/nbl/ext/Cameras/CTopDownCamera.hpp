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

    CTopDownCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, const SWorldFrame& frame = SWorldFrame::yUpLeftHanded())
        : base_t(position, target, frame)
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
            distance <= ZeroTolerance)
        {
            return false;
        }

        m_orbit.distance = std::clamp(distance, MinDistance, MaxDistance);
        m_orbit.angles.x = resolveTopDownYaw(pose.orientation, getFrame(), m_orbit.angles.x);
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
    /// @brief Recover the azimuth an orientation encodes, falling back when the pose carries none.
    static inline double resolveTopDownYaw(const hlsl::math::quaternion<hlsl::float64_t>& orientation, const SWorldFrame& frame, const double fallbackYaw)
    {
        const auto basis = CCameraMathUtilities::basisFromOrientation(orientation, frame.handedness);
        // a camera looking at its target from azimuth `yaw` looks along -(forward * cos(yaw) + right * sin(yaw)) in
        // the horizon plane; straight above the target that part vanishes and its up, the orbit tangent
        // `upHintFromYawPitch(yaw, 90 deg)` = -forward * cos(yaw) - right * sin(yaw), carries the same azimuth
        auto horizon = basis.forward - frame.up * hlsl::dot(basis.forward, frame.up);
        if (CCameraMathUtilities::isNearlyZeroVector(horizon, ZeroTolerance))
            horizon = basis.up - frame.up * hlsl::dot(basis.up, frame.up);
        if (CCameraMathUtilities::isNearlyZeroVector(horizon, ZeroTolerance))
            return fallbackYaw;
        return hlsl::atan2(-hlsl::dot(horizon, frame.right), -hlsl::dot(horizon, frame.forward));
    }

public:
    /// @brief Elevation of the camera above its target, in radians: straight above it, looking down.
    static inline constexpr double TopDownPitch = 90.0 * (hlsl::numbers::pi<double> / 180.0);
};

}

#endif
