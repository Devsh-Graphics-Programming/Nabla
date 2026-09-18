// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Free-position camera that allows full yaw/pitch/roll rotation.
///
/// Controls: `translate` in the camera's own frame (x right, y up, z forward), world units. `rotate.x`,
/// `rotate.y` and `rotate.z` pitch, yaw and roll about the camera's own right, up and forward axes, radians,
/// unclamped.
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

    virtual uint32_t getAcceptedControls() const override
    {
        return AcceptedControls;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Free;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Free-Look Camera";
    }

    static inline constexpr uint32_t AcceptedControls = ECameraControlAxis::Translate | ECameraControlAxis::Rotate;

protected:
    virtual bool applyControls(const SCameraControls& controls) override
    {
        // a copy, because the pose is replaced below while it still anchors the rotation axes and the translation
        const auto anchor = m_gimbal.getPose();
        const auto anchorBasis = m_gimbal.getBasis();

        // rotations about the anchor's own axes, translation in the anchor frame
        const auto pitch = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.right, controls.rotate.x);
        const auto yaw = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.up, controls.rotate.y);
        const auto roll = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.forward, controls.rotate.z);
        const auto newPosition = anchor.position + anchor.orientation.transformVector(controls.translate, true);

        return m_gimbal.setPose(SCameraRigPose{
            .position = newPosition,
            .orientation = yaw * pitch * roll * anchor.orientation
        });
    }

private:
    CCameraGimbal m_gimbal;
};

}

#endif // _C_FREE_CAMERA_HPP_
