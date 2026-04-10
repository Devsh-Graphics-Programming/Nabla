// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_MANIPULATION_UTILITIES_HPP_
#define _C_CAMERA_MANIPULATION_UTILITIES_HPP_

#include <algorithm>
#include <vector>

#include "CCameraPresetFlow.hpp"
#include "CCameraVirtualEventUtilities.hpp"

namespace nbl::core
{

struct SCameraConstraintDefaults final
{
    static constexpr float PitchMinDeg = -80.0f;
    static constexpr float PitchMaxDeg = 80.0f;
    static constexpr float YawMinDeg = -180.0f;
    static constexpr float YawMaxDeg = 180.0f;
    static constexpr float RollMinDeg = -180.0f;
    static constexpr float RollMaxDeg = 180.0f;
    static constexpr float MinDistance = SCameraTargetRelativeTraits::MinDistance;
    static constexpr float MaxDistance = SCameraTargetRelativeTraits::DefaultMaxDistance;
};

/// @brief Reusable constraint settings for post-manipulation camera clamping.
struct SCameraConstraintSettings
{
    bool enabled = false;
    bool clampPitch = false;
    bool clampYaw = false;
    bool clampRoll = false;
    bool clampDistance = false;
    float pitchMinDeg = SCameraConstraintDefaults::PitchMinDeg;
    float pitchMaxDeg = SCameraConstraintDefaults::PitchMaxDeg;
    float yawMinDeg = SCameraConstraintDefaults::YawMinDeg;
    float yawMaxDeg = SCameraConstraintDefaults::YawMaxDeg;
    float rollMinDeg = SCameraConstraintDefaults::RollMinDeg;
    float rollMaxDeg = SCameraConstraintDefaults::RollMaxDeg;
    float minDistance = SCameraConstraintDefaults::MinDistance;
    float maxDistance = SCameraConstraintDefaults::MaxDistance;
};

struct CCameraManipulationUtilities final
{
public:
    /// @brief Apply an authored world-space reference frame through the shared camera runtime entry point.
    static inline bool applyReferenceFrameToCamera(ICamera* camera, const hlsl::float64_t4x4& referenceFrame)
    {
        if (!camera)
            return false;

        return camera->manipulateWithUnitMotionScales({}, &referenceFrame);
    }

    /// @brief Scale translation and rotation event magnitudes without touching unrelated event types.
    static inline void scaleVirtualEvents(std::vector<CVirtualGimbalEvent>& events, const uint32_t count, const float translationScale, const float rotationScale)
    {
        for (uint32_t i = 0u; i < count; ++i)
        {
            auto& ev = events[i];
            if (CVirtualGimbalEvent::isTranslationEvent(ev.type))
            {
                ev.magnitude *= translationScale;
            }
            else if (CVirtualGimbalEvent::isRotationEvent(ev.type))
            {
                ev.magnitude *= rotationScale;
            }
        }
    }

    /// @brief Reinterpret world-space translation intents as local camera-space movement events.
    static inline void remapTranslationEventsFromWorldToCameraLocal(ICamera* camera, std::vector<CVirtualGimbalEvent>& events, uint32_t& count)
    {
        if (!camera)
            return;

        std::vector<CVirtualGimbalEvent> filtered;
        filtered.reserve(events.size());

        for (uint32_t i = 0u; i < count; ++i)
        {
            const auto& ev = events[i];
            if (!CVirtualGimbalEvent::isTranslationEvent(ev.type))
                filtered.emplace_back(ev);
        }

        const auto worldDelta = CCameraVirtualEventUtilities::collectSignedTranslationDelta({ events.data(), count });
        if (hlsl::CCameraMathUtilities::isNearlyZeroVector(worldDelta, static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon)))
        {
            events = std::move(filtered);
            count = static_cast<uint32_t>(events.size());
            return;
        }

        CCameraVirtualEventUtilities::appendWorldTranslationAsLocalEvents(filtered, camera->getGimbal().getOrientation(), worldDelta);

        events = std::move(filtered);
        count = static_cast<uint32_t>(events.size());
    }

    /// @brief Apply shared distance and Euler-angle constraints after manipulation.
    static inline bool applyCameraConstraints(const CCameraGoalSolver& solver, ICamera* camera, const SCameraConstraintSettings& constraints)
    {
        if (!constraints.enabled || !camera)
            return false;

        if (camera->hasCapability(ICamera::SphericalTarget))
        {
            if (!constraints.clampDistance)
                return false;

            ICamera::SphericalTargetState sphericalState;
            if (!camera->tryGetSphericalTargetState(sphericalState))
                return false;

            const float clamped = std::clamp<float>(sphericalState.distance, constraints.minDistance, constraints.maxDistance);
            if (clamped == sphericalState.distance)
                return false;

            return camera->trySetSphericalDistance(clamped);
        }

        if (!(constraints.clampPitch || constraints.clampYaw || constraints.clampRoll))
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto pos = gimbal.getPosition();
        const auto eulerDeg = hlsl::CCameraMathUtilities::getCameraOrientationEulerDegrees(gimbal.getOrientation());

        auto clamped = eulerDeg;
        if (constraints.clampPitch)
            clamped.x = std::clamp(clamped.x, static_cast<decltype(clamped.x)>(constraints.pitchMinDeg), static_cast<decltype(clamped.x)>(constraints.pitchMaxDeg));
        if (constraints.clampYaw)
            clamped.y = std::clamp(clamped.y, static_cast<decltype(clamped.y)>(constraints.yawMinDeg), static_cast<decltype(clamped.y)>(constraints.yawMaxDeg));
        if (constraints.clampRoll)
            clamped.z = std::clamp(clamped.z, static_cast<decltype(clamped.z)>(constraints.rollMinDeg), static_cast<decltype(clamped.z)>(constraints.rollMaxDeg));

        if (clamped.x == eulerDeg.x && clamped.y == eulerDeg.y && clamped.z == eulerDeg.z)
            return false;

        CCameraPreset preset;
        preset.goal.position = pos;
        preset.goal.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(clamped);
        return CCameraPresetFlowUtilities::applyPreset(solver, camera, preset);
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_MANIPULATION_UTILITIES_HPP_

