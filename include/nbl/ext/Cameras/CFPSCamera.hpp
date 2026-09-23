// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FPS_CAMERA_HPP_
#define _C_FPS_CAMERA_HPP_

#include <cmath>

#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Free-position camera with world-space translation and yaw/pitch rotation.
///
/// Holding roll at zero is this rig's contract, not a missing feature: the state is a position plus an
/// orientation rebuilt every step from yaw and a pitch bounded short of straight up and down, which is
/// what keeps the horizon level. Use `CFreeCamera` when you want the roll.
///
/// Controls: `translate` in the camera's own frame (x right, y up, z forward), world units. `rotate.x` pitch,
/// clamped short of straight up and down; `rotate.y` yaw about the world up axis; radians.
class CFPSCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFPSCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity())
        : base_t(), m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation })
    {
        const auto pitchYawRoll = CCameraMathUtilities::getPitchYawRollRadians(m_gimbal.getOrientation());
        m_gimbal.setOrientation(hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(pitchYawRoll.y, pitchYawRoll.x, 0.0));
    }
	~CFPSCamera() = default;

    const CCameraGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    using base_t::setPose;

    /// @brief Place the rig at the position of `pose` and at the pitch and yaw its orientation encodes.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        if (!CCameraMathUtilities::isFiniteVec3(pose.position))
            return false;

        const auto pitchYawRoll = CCameraMathUtilities::getPitchYawRollRadians(pose.orientation);
        if (!CCameraMathUtilities::isFiniteScalar(pitchYawRoll.x) || !CCameraMathUtilities::isFiniteScalar(pitchYawRoll.y))
            return false;

        const auto pitch = std::clamp<hlsl::float64_t>(pitchYawRoll.x, MinVerticalAngle, MaxVerticalAngle);
        m_gimbal.setPose(SCameraRigPose{
            .position = pose.position,
            .orientation = hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(pitchYawRoll.y, pitch, 0.0)
        });

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
        // a copy, because the pose is replaced below while its orientation still anchors the translation
        const auto anchor = m_gimbal.getPose();
        const auto pitchYawRoll = CCameraMathUtilities::getPitchYawRollRadians(m_gimbal.getOrientation());

        const auto newPitch = std::clamp<hlsl::float64_t>(pitchYawRoll.x + controls.rotate.x, MinVerticalAngle, MaxVerticalAngle);
        const auto newYaw = pitchYawRoll.y + controls.rotate.y;

        // the translation is applied in the anchor frame, so it is resolved before the pose is replaced
        const auto newPosition = anchor.position + anchor.orientation.transformVector(controls.translate, true);
        return m_gimbal.setPose(SCameraRigPose{
            .position = newPosition,
            .orientation = hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(newYaw, newPitch, 0.0)
        });
    }

public:
    /// @brief Pitch limit in radians. It stops 2 deg short of straight up and down.
    static inline constexpr hlsl::float64_t MaxVerticalAngle = 88.0 * (hlsl::numbers::pi<hlsl::float64_t> / 180.0);
    static inline constexpr hlsl::float64_t MinVerticalAngle = -MaxVerticalAngle;

private:
    CCameraGimbal m_gimbal;
};

}

#endif // _C_FPS_CAMERA_HPP_

