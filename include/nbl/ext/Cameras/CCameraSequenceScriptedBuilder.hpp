// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_

#include <string>

#include "CCameraScriptedRuntime.hpp"
#include "CCameraSequenceScript.hpp"
#include "ICamera.hpp"

namespace nbl::system
{

/// @brief Build expanded scripted runtime data from a compiled camera-sequence segment.
///
/// The builder converts compiled sequence frames into the shared runtime event
/// and check payloads used by camera-sequence consumers.
struct CCameraSequenceScriptedSegmentBuildInfo
{
    /// @brief Planar index that receives the compiled segment.
    uint32_t planarIx = 0u;
    /// @brief Number of windows the consumer can actually route presentation actions to.
    size_t availableWindowCount = 1u;
    /// @brief Whether secondary-window presentation actions are emitted.
    bool useWindow = false;
    /// @brief Whether per-frame follow-lock checks are generated for this segment.
    bool includeFollowTargetLock = false;
};

struct CCameraSequenceScriptedBuilderUtilities final
{
    /// @brief Append one compiled segment as expanded scripted runtime payloads.
    static inline bool appendCompiledSequenceSegmentToScriptedTimeline(
        CCameraScriptedTimeline& timeline,
        const uint64_t baseFrame,
        const core::CCameraSequenceCompiledSegment& compiledSegment,
        const CCameraSequenceScriptedSegmentBuildInfo& buildInfo,
        std::string* error = nullptr)
    {
        std::vector<core::CCameraSequenceCompiledFramePolicy> framePolicies;
        if (!core::CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, buildInfo.includeFollowTargetLock))
        {
            if (error)
                *error = "Failed to build compiled frame policies.";
            return false;
        }

        CCameraScriptedRuntimeUtilities::appendScriptedSegmentLabelEvent(timeline, baseFrame, compiledSegment.name);
        CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, 0);
        CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
        if (!compiledSegment.presentations.empty())
        {
            CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[0].projection));
            CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded, compiledSegment.presentations[0].leftHanded ? 1 : 0);
        }
        if (compiledSegment.resetCamera)
            CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::ResetActiveCamera, 1);

        if (buildInfo.useWindow)
        {
            for (size_t windowIx = 1u; windowIx < std::min(compiledSegment.presentations.size(), buildInfo.availableWindowCount); ++windowIx)
            {
                CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, static_cast<int32_t>(windowIx));
                CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
                CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[windowIx].projection));
                CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded, compiledSegment.presentations[windowIx].leftHanded ? 1 : 0);
            }
            CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, 0);
        }

        for (const auto& policy : framePolicies)
        {
            core::CCameraPreset preset;
            if (!core::CCameraKeyframeTrackUtilities::tryBuildKeyframeTrackPresetAtTime(compiledSegment.track, policy.sampleTime, preset))
            {
                if (error)
                    *error = "Failed to sample compiled segment track.";
                return false;
            }
            CCameraScriptedRuntimeUtilities::appendScriptedGoalEvent(
                timeline,
                baseFrame + policy.frameOffset,
                core::CCameraPresetUtilities::makeGoalFromPreset(preset));

            if (compiledSegment.usesTrackedTargetTrack())
            {
                core::CCameraSequenceTrackedTargetPose trackedTargetPose;
                if (!core::CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, policy.sampleTime, trackedTargetPose))
                {
                    if (error)
                        *error = "Failed to sample compiled tracked-target track.";
                    return false;
                }

                core::ICamera::CGimbal gimbal({ .position = trackedTargetPose.position, .orientation = trackedTargetPose.orientation });
                CCameraScriptedRuntimeUtilities::appendScriptedTrackedTargetTransformEvent(timeline, baseFrame + policy.frameOffset, gimbal.operator()<hlsl::float64_t4x4>());
            }

            if (policy.baseline)
                CCameraScriptedRuntimeUtilities::appendScriptedBaselineCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.continuityStep)
            {
                CCameraScriptedRuntimeUtilities::appendScriptedGimbalStepCheck(
                    timeline,
                    baseFrame + policy.frameOffset,
                    compiledSegment.continuity.hasPosDeltaConstraint,
                    compiledSegment.continuity.maxPosDelta,
                    compiledSegment.continuity.minPosDelta,
                    compiledSegment.continuity.hasEulerDeltaConstraint,
                    compiledSegment.continuity.maxEulerDeltaDeg,
                    compiledSegment.continuity.minEulerDeltaDeg);
            }
            if (policy.followTargetLock)
                CCameraScriptedRuntimeUtilities::appendScriptedFollowTargetLockCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.capture)
                timeline.captureFrames.emplace_back(baseFrame + policy.frameOffset);
        }

        return true;
    }
};

} // namespace nbl::system

#endif
