// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Free-position camera that allows full yaw/pitch/roll rotation.
class CFreeCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFreeCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity())
        : base_t(), m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation }) {}
    ~CFreeCamera() = default;

    const CCameraGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    using base_t::setPose;

    /// @brief Place the rig at `pose` verbatim.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        if (!CCameraMathUtilities::isFiniteVec3(pose.position) || !CCameraMathUtilities::isFiniteQuaternion(pose.orientation))
            return false;

        m_gimbal.setPose(pose);
        return true;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = accumulateVirtualEvents<AllowedVirtualEvents>(virtualEvents);
        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);

        // a copy, because the pose is replaced below while it still anchors the rotation axes and the translation
        const auto anchor = m_gimbal.getPose();
        const auto anchorBasis = m_gimbal.getBasis();

        // rotations about the anchor's own axes, translation in the anchor frame
        const auto pitch = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.right, deltaRotation.x);
        const auto yaw = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.up, deltaRotation.y);
        const auto roll = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.forward, deltaRotation.z);
        const auto newPosition = anchor.position + anchor.orientation.transformVector(hlsl::float64_t3(deltaTranslation), true);

        return m_gimbal.setPose(SCameraRigPose{
            .position = newPosition,
            .orientation = yaw * pitch * roll * anchor.orientation
        });
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Free;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Free-Look Camera";
    }

private:
    CCameraGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
};

}

#endif // _C_FREE_CAMERA_HPP_
