// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Free-position camera that allows full yaw/pitch/roll rotation.
class CFreeCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFreeCamera(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>())
        : base_t(), m_gimbal(typename base_t::CGimbal::base_t::SCreationParameters{ .position = position, .orientation = orientation }) {}
    ~CFreeCamera() = default;

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        CReferenceTransform reference;
        if (not m_gimbal.extractReferenceTransform(&reference, referenceFrame))
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        m_gimbal.begin();
        {
            const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
            const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
            const auto pitch = hlsl::CCameraMathUtilities::makeQuaternionFromAxisAngle(hlsl::normalize(hlsl::float64_t3(reference.frame[0])), deltaRotation.x);
            const auto yaw = hlsl::CCameraMathUtilities::makeQuaternionFromAxisAngle(hlsl::normalize(hlsl::float64_t3(reference.frame[1])), deltaRotation.y);
            const auto roll = hlsl::CCameraMathUtilities::makeQuaternionFromAxisAngle(hlsl::normalize(hlsl::float64_t3(reference.frame[2])), deltaRotation.z);

            m_gimbal.setOrientation(hlsl::CCameraMathUtilities::normalizeQuaternion(yaw * pitch * roll * reference.orientation));
            m_gimbal.setPosition(hlsl::float64_t3(reference.frame[3]) + hlsl::CCameraMathUtilities::rotateVectorByQuaternion(reference.orientation, hlsl::float64_t3(deltaTranslation)));
        }
        m_gimbal.end();

        manipulated &= bool(m_gimbal.getManipulationCounter());

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
