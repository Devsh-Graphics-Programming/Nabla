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
        : base_t(), m_gimbal(typename base_t::CGimbal::base_t::SCreationParameters{ .position = position, .orientation = orientation }) {}
    ~CFreeCamera() = default;

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    using base_t::setPose;

    /// @brief Place the rig at `pose` verbatim.
    virtual bool setPose(const SCameraRigPose& pose) override
    {
        if (!CCameraMathUtilities::isFiniteVec3(pose.position) || !CCameraMathUtilities::isFiniteQuaternion(pose.orientation))
            return false;

        m_gimbal.begin();
        {
            m_gimbal.setOrientation(pose.orientation);
            m_gimbal.setPosition(pose.position);
        }
        m_gimbal.end();
        m_gimbal.updateView();

        return true;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (virtualEvents.empty())
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);

        // copies, because `setOrientation` rewrites the gimbal's orientation and basis in place
        const auto anchorPosition = m_gimbal.getPosition();
        const auto anchorOrientation = m_gimbal.getOrientation();
        const auto anchorBasis = m_gimbal.getBasis();

        const auto pitch = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.right, deltaRotation.x);
        const auto yaw = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.up, deltaRotation.y);
        const auto roll = hlsl::math::quaternion<hlsl::float64_t>::createFromAxisAngle(anchorBasis.forward, deltaRotation.z);

        m_gimbal.begin();
        {
            m_gimbal.setOrientation(hlsl::normalize(yaw * pitch * roll * anchorOrientation));
            m_gimbal.setPosition(anchorPosition + hlsl::normalize(anchorOrientation).transformVector(hlsl::float64_t3(deltaTranslation), true));
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());

        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
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
    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
};

}

#endif // _C_FREE_CAMERA_HPP_
