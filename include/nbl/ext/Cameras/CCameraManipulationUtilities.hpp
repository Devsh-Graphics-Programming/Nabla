// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_MANIPULATION_UTILITIES_HPP_
#define _C_CAMERA_MANIPULATION_UTILITIES_HPP_

#include <algorithm>
#include <vector>

#include "CCameraVirtualEventUtilities.hpp"

namespace nbl::core
{

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
};

} // namespace nbl::core

#endif // _C_CAMERA_MANIPULATION_UTILITIES_HPP_

