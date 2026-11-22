// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FPS_CAMERA_HPP_
#define _C_FPS_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Free-position camera with camera-relative translation and yaw/pitch rotation.
///
/// Holding roll at zero is this rig's contract, not a missing feature: the state is a position plus a yaw and a
/// pitch, from which the orientation is rebuilt every step with the frame's up as the up hint, which is what keeps
/// the horizon level. Use `CFreeCamera` when you want the roll.
///
/// The two angles are measured in the camera's `SWorldFrame`: yaw 0 and pitch 0 look along the frame's forward,
/// positive yaw turns the look direction toward the frame's right about its up, positive pitch lifts it toward the
/// up. See `directionFromYawPitch`.
///
/// Controls: `translate` in the camera's own frame (x right, y up, z forward), world units. `rotate.y` turns the
/// look direction toward the camera's right, `rotate.x` toward its down, radians; the pitch is clamped short of
/// straight up and down.
class CFPSCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFPSCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity(), const SWorldFrame& frame = SWorldFrame::yUpLeftHanded())
        : base_t(frame), m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation })
    {
        setPose(m_gimbal.getPose());
    }
	~CFPSCamera() = default;

    const CCameraGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    using base_t::setPose;

    /// @brief Place the rig at the position of `pose` and at the yaw and pitch of its look direction; roll is dropped.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        if (!CCameraMathUtilities::isFiniteVec3(pose.position) || !CCameraMathUtilities::isFiniteQuaternion(pose.orientation))
            return false;

        const auto& frame = getFrame();
        const auto forward = CCameraMathUtilities::basisFromOrientation(pose.orientation, frame.handedness).forward;
        const auto yawPitch = CCameraMathUtilities::yawPitchFromDirection(frame.right, frame.up, frame.forward, forward);

        m_yaw = yawPitch.x;
        m_pitch = std::clamp<hlsl::float64_t>(yawPitch.y, MinVerticalAngle, MaxVerticalAngle);
        rebuildGimbal(pose.position);
        return true;
    }

    virtual uint32_t getAcceptedControls() const override
    {
        return AcceptedControls;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::FPS;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "FPS Camera";
    }

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::Translate | ECameraControlAxis::RotateX | ECameraControlAxis::RotateY;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        // the translation is resolved in the basis the controls were issued in, before the rotation changes it
        const auto basis = getBasis();
        const auto newPosition = m_gimbal.getPosition() + CCameraMathUtilities::transformLocalVectorToWorldBasis(controls.translate, basis.right, basis.up, basis.forward);

        m_yaw += controls.rotate.y;
        // `rotate.x` tilts toward down, so it lowers the elevation
        m_pitch = std::clamp<hlsl::float64_t>(m_pitch - controls.rotate.x, MinVerticalAngle, MaxVerticalAngle);
        return rebuildGimbal(newPosition);
    }

    /// @brief Write the pose for the current yaw and pitch at `position` into the gimbal.
    bool rebuildGimbal(const hlsl::float64_t3& position)
    {
        const auto& frame = getFrame();
        const auto forward = CCameraMathUtilities::directionFromYawPitch(frame.right, frame.up, frame.forward, m_yaw, m_pitch);
        // the up hint is the frame's up, so the result has no roll; the pitch clamp keeps forward away from it
        SCameraBasis<hlsl::float64_t> basis;
        CCameraMathUtilities::tryBuildBasis(forward, frame.up, frame, basis);
        return m_gimbal.setPose(SCameraRigPose{
            .position = position,
            .orientation = CCameraMathUtilities::orientationFromBasis(basis, frame.handedness)
        });
    }

public:
    /// @brief Pitch limit in radians. It stops 2 deg short of straight up and down.
    static inline constexpr hlsl::float64_t MaxVerticalAngle = 88.0 * (hlsl::numbers::pi<hlsl::float64_t> / 180.0);
    static inline constexpr hlsl::float64_t MinVerticalAngle = -MaxVerticalAngle;

private:
    CCameraGimbal m_gimbal;
    /// @brief Yaw in `getFrame()`, radians, unbounded; `rotate.y` adds to it.
    hlsl::float64_t m_yaw = 0.0;
    /// @brief Elevation in radians within [MinVerticalAngle, MaxVerticalAngle]; `rotate.x` subtracts from it.
    hlsl::float64_t m_pitch = 0.0;
};

}

#endif // _C_FPS_CAMERA_HPP_
