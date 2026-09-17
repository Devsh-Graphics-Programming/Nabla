// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
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
class CFPSCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFPSCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity())
        : base_t(), m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation })
    {
        const auto pitchYaw = CCameraMathUtilities::getPitchYawFromForwardVector(m_gimbal.getForward());
        m_gimbal.setOrientation(CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(hlsl::float64_t3(pitchYaw.x, pitchYaw.y, 0.0)));
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

        const auto pitchYaw = CCameraMathUtilities::getPitchYawFromOrientation(pose.orientation);
        if (!CCameraMathUtilities::isFiniteScalar(pitchYaw.x) || !CCameraMathUtilities::isFiniteScalar(pitchYaw.y))
            return false;

        const auto pitch = std::clamp<hlsl::float64_t>(pitchYaw.x, MinVerticalAngle, MaxVerticalAngle);
        m_gimbal.setPose(SCameraRigPose{
            .position = pose.position,
            .orientation = CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(hlsl::float64_t3(pitch, pitchYaw.y, 0.0))
        });

        return true;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);

        // a copy, because the pose is replaced below while its orientation still anchors the translation
        const auto anchor = m_gimbal.getPose();
        const auto pitchYaw = CCameraMathUtilities::getPitchYawFromForwardVector(m_gimbal.getForward());

        const auto newPitch = std::clamp<hlsl::float64_t>(pitchYaw.x + scaleVirtualRotation(impulse.dVirtualRotation.x), MinVerticalAngle, MaxVerticalAngle);
        const auto newYaw = pitchYaw.y + scaleVirtualRotation(impulse.dVirtualRotation.y);

        // the translation is applied in the anchor frame, so it is resolved before the pose is replaced
        const auto newPosition = anchor.position + anchor.orientation.transformVector(hlsl::float64_t3(deltaTranslation), true);
        return m_gimbal.setPose(SCameraRigPose{
            .position = newPosition,
            .orientation = CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(hlsl::float64_t3(newPitch, newYaw, 0.0))
        });
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::FPS;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "FPS Camera";
    }

private:

    CCameraGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr hlsl::float64_t MaxVerticalAngle = SCameraViewRigDefaults::FpsVerticalPitchLimitRad;
    static inline constexpr hlsl::float64_t MinVerticalAngle = -MaxVerticalAngle;
};

}

#endif // _C_FPS_CAMERA_HPP_

