// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::ext::cameras
{

/// @brief Free-position camera that allows full yaw/pitch/roll rotation.
///
/// Controls: `translate` in the camera's own frame (x right, y up, z forward), world units. Rotations about the
/// camera's own axes, radians, unclamped: `rotate.z` rolls right toward up, `rotate.x` pitches forward toward
/// down, `rotate.y` yaws forward toward right, applied in that order.
class CFreeCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFreeCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity(), const SWorldFrame& frame = SWorldFrame::yUpLeftHanded())
        : base_t(frame), m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation }) {}
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
        // the translation is resolved in the basis the controls were issued in, before the rotation changes it
        const auto basis = getBasis();
        const auto newPosition = m_gimbal.getPosition() + CCameraMathUtilities::transformLocalVectorToWorldBasis(controls.translate, basis.right, basis.up, basis.forward);

        // each rotation turns two of the axes toward each other inside their plane
        auto right = basis.right;
        auto up = basis.up;
        auto forward = basis.forward;
        // roll about forward: right toward up
        {
            const auto rolledRight = CCameraMathUtilities::rotateTowards(right, up, controls.rotate.z);
            up = CCameraMathUtilities::rotateTowards(up, -right, controls.rotate.z);
            right = rolledRight;
        }
        // pitch about right: forward toward down
        {
            const auto pitchedForward = CCameraMathUtilities::rotateTowards(forward, -up, controls.rotate.x);
            up = CCameraMathUtilities::rotateTowards(up, forward, controls.rotate.x);
            forward = pitchedForward;
        }
        // yaw about up: forward toward right
        forward = CCameraMathUtilities::rotateTowards(forward, right, controls.rotate.y);

        // re-orthonormalized, with right rebuilt for the handedness
        SCameraBasis<hlsl::float64_t> newBasis;
        if (!CCameraMathUtilities::tryBuildBasis(forward, up, getFrame(), newBasis))
            return false;

        return m_gimbal.setPose(SCameraRigPose{
            .position = newPosition,
            .orientation = CCameraMathUtilities::orientationFromBasis(newBasis, getHandedness())
        });
    }

private:
    CCameraGimbal m_gimbal;
};

}

#endif // _C_FREE_CAMERA_HPP_
