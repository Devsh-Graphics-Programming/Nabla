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
/// The runtime state consists of position plus an upright orientation derived
/// from yaw and pitch. Reference-frame application rejects arbitrary roll and
/// rebuilds the legal FPS orientation from the extracted forward axis.
class CFPSCamera final : public ICamera
{ 
public:
    using base_t = ICamera;
    struct SFpsCameraDefaults final
    {
        static inline constexpr float RollValidationEpsilonDeg = 1.e-4f;
        static inline constexpr float StraightRollDeg = 0.0f;
        static inline constexpr float InvertedRollDeg = 180.0f;
    };

    CFPSCamera(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity())
        : base_t(), m_gimbal(typename base_t::CGimbal::base_t::SCreationParameters{ .position = position, .orientation = orientation }) 
    {
        m_gimbal.begin();
        {
            const auto pitchYaw = CCameraMathUtilities::getPitchYawFromForwardVector(m_gimbal.getZAxis());
            m_gimbal.setOrientation(CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(hlsl::float64_t3(pitchYaw.x, pitchYaw.y, 0.0)));
        }
        m_gimbal.end();
    }
	~CFPSCamera() = default;

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

        auto validateReference = [&]()
        {
            if (referenceFrame)
            {
                const float roll = static_cast<float>(hlsl::degrees(CCameraMathUtilities::getQuaternionEulerRadiansYXZ(reference.orientation).z));
                const bool matchesStraightRoll =
                    CCameraMathUtilities::getWrappedAngleDistanceDegrees(roll, SFpsCameraDefaults::StraightRollDeg) <= SFpsCameraDefaults::RollValidationEpsilonDeg;
                const bool matchesInvertedRoll =
                    CCameraMathUtilities::getWrappedAngleDistanceDegrees(roll, SFpsCameraDefaults::InvertedRollDeg) <= SFpsCameraDefaults::RollValidationEpsilonDeg;

                if (!(matchesStraightRoll || matchesInvertedRoll))
                    return false;
            }

            return true;
        };

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        m_gimbal.begin();
        {
            const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
            const auto pitchYaw = CCameraMathUtilities::getPitchYawFromForwardVector(reference.getBasis().forward);
            const float newPitch = std::clamp<float>(static_cast<float>(pitchYaw.x + scaleVirtualRotation(impulse.dVirtualRotation.x)), MinVerticalAngle, MaxVerticalAngle);
            const float newYaw = static_cast<float>(pitchYaw.y + scaleVirtualRotation(impulse.dVirtualRotation.y));

            if (validateReference())
                m_gimbal.setOrientation(CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(hlsl::float64_t3(newPitch, newYaw, 0.0f)));
            m_gimbal.setPosition(reference.getPosition() + hlsl::normalize(reference.orientation).transformVector(hlsl::float64_t3(deltaTranslation), true));
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
        return CameraKind::FPS;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "FPS Camera";
    }

private:

    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr float MaxVerticalAngle = static_cast<float>(SCameraViewRigDefaults::FpsVerticalPitchLimitRad);
    static inline constexpr float MinVerticalAngle = -MaxVerticalAngle;
};

}

#endif // _C_FPS_CAMERA_HPP_

