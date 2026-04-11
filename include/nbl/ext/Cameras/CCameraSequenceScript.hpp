// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPT_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "CCameraMathUtilities.hpp"
#include "CCameraKeyframeTrack.hpp"
#include "CCameraPathUtilities.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "IPlanarProjection.hpp"

namespace nbl::core
{

/// @brief Compact authored camera-sequence format shared by playback, scripting, and validation helpers.
///
/// The authored file describes:
///
/// - which camera kind a segment targets
/// - which reusable projection presentations are shown
/// - which keyframed camera goals are sampled over time
/// - which tracked-target poses are sampled over time
/// - which continuity thresholds and capture points are generated
///
/// The format does not store:
///
/// - per-frame low-level event dumps
/// - runtime-specific window actions as authored source data
/// - ImGuizmo transforms as the primary authored primitive
///
/// Consumers may expand the compact sequence into runtime events and per-frame
/// checks. The authored data remains camera-domain data and is not a device- or
/// UI-specific event dump.

/// @brief Authored projection view request for camera-sequence playback.
struct CCameraSequencePresentation
{
    IPlanarProjection::CProjection::ProjectionType projection = IPlanarProjection::CProjection::Perspective;
    bool leftHanded = true;
};

/// @brief Shared continuity thresholds authored once and reused per sequence segment.
/// Max bounds are enforced per-step, while minimum progress can be satisfied by either position or rotation change.
struct CCameraSequenceContinuitySettings
{
    bool baseline = true;
    bool step = true;
    bool hasPosDeltaConstraint = true;
    float minPosDelta = 0.00025f;
    float maxPosDelta = 2.f;
    bool hasEulerDeltaConstraint = false;
    float minEulerDeltaDeg = 0.f;
    float maxEulerDeltaDeg = 1.f;
};

/// @brief Relative goal adjustment authored against an initial preset captured from the target camera.
/// Deltas stay camera-domain and avoid binding the authored file to any specific input device or consumer.
struct CCameraSequenceGoalDelta
{
    struct SOrbitDelta final
    {
        hlsl::float64_t2 uvDeltaRad = hlsl::float64_t2(0.0);
        float distanceDelta = 0.f;
        bool hasU = false;
        bool hasV = false;
        bool hasDistance = false;

        inline bool hasAny() const
        {
            return hasU || hasV || hasDistance;
        }

        inline void setUDeltaDeg(const double valueDeg)
        {
            uvDeltaRad.x = static_cast<hlsl::float64_t>(hlsl::radians(valueDeg));
            hasU = true;
        }

        inline void setVDeltaDeg(const double valueDeg)
        {
            uvDeltaRad.y = static_cast<hlsl::float64_t>(hlsl::radians(valueDeg));
            hasV = true;
        }

        inline void setDistanceDelta(const float valueScalar)
        {
            distanceDelta = valueScalar;
            hasDistance = true;
        }
    };

    struct SPathDelta final
    {
        SCameraPathDelta value = {};
        bool hasS = false;
        bool hasU = false;
        bool hasV = false;
        bool hasRoll = false;

        inline bool hasAny() const
        {
            return hasS || hasU || hasV || hasRoll;
        }

        inline void setSDeltaDeg(const double valueDeg)
        {
            value.s = static_cast<hlsl::float64_t>(hlsl::radians(valueDeg));
            hasS = true;
        }

        inline void setUDelta(const double valueScalar)
        {
            value.u = static_cast<hlsl::float64_t>(valueScalar);
            hasU = true;
        }

        inline void setVDelta(const double valueScalar)
        {
            value.v = static_cast<hlsl::float64_t>(valueScalar);
            hasV = true;
        }

        inline void setRollDeltaDeg(const double valueDeg)
        {
            value.roll = static_cast<hlsl::float64_t>(hlsl::radians(valueDeg));
            hasRoll = true;
        }

        inline SCameraPathDelta buildAppliedDelta() const
        {
            SCameraPathDelta delta = {};
            if (hasS)
                delta.s = value.s;
            if (hasU)
                delta.u = value.u;
            if (hasV)
                delta.v = value.v;
            if (hasRoll)
                delta.roll = value.roll;
            return delta;
        }
    };

    bool hasPositionOffset = false;
    hlsl::float64_t3 positionOffset = hlsl::float64_t3(0.0);

    bool hasRotationEulerDegOffset = false;
    hlsl::float32_t3 rotationEulerDegOffset = hlsl::float32_t3(0.f);

    bool hasTargetOffset = false;
    hlsl::float64_t3 targetOffset = hlsl::float64_t3(0.0);

    SOrbitDelta orbitDelta = {};

    SPathDelta pathDelta = {};

    bool hasDynamicBaseFovDelta = false;
    float dynamicBaseFovDelta = 0.f;

    bool hasDynamicReferenceDistanceDelta = false;
    float dynamicReferenceDistanceDelta = 0.f;
};

/// @brief One authored keyframe inside a reusable camera-sequence segment.
/// A keyframe can be described either as an absolute preset or as a delta relative to the captured reference preset.
struct CCameraSequenceKeyframe
{
    float time = 0.f;
    bool hasAbsolutePreset = false;
    CCameraPreset absolutePreset = {};
    bool hasDelta = false;
    CCameraSequenceGoalDelta delta = {};
};

/// @brief Concrete tracked-target pose sampled from a shared authored sequence.
struct CCameraSequenceTrackedTargetPose final : SCameraRigPose
{
};

/// @brief Relative tracked-target adjustment authored against an initial tracked-target pose.
struct CCameraSequenceTrackedTargetDelta
{
    bool hasPositionOffset = false;
    hlsl::float64_t3 positionOffset = hlsl::float64_t3(0.0);

    bool hasRotationEulerDegOffset = false;
    hlsl::float32_t3 rotationEulerDegOffset = hlsl::float32_t3(0.f);
};

/// @brief One authored tracked-target keyframe inside a reusable camera-sequence segment.
/// Target keyframes stay camera-domain and can drive follow behavior without runtime-object references.
struct CCameraSequenceTrackedTargetKeyframe
{
    float time = 0.f;
    bool hasAbsolutePosition = false;
    hlsl::float64_t3 absolutePosition = hlsl::float64_t3(0.0);
    bool hasAbsoluteRotationEulerDeg = false;
    hlsl::float32_t3 absoluteRotationEulerDeg = hlsl::float32_t3(0.f);
    bool hasDelta = false;
    CCameraSequenceTrackedTargetDelta delta = {};
};

/// @brief Runtime sampled tracked-target track built from an authored segment plus a reference pose.
/// Keyframes are normalized by time before sampling. Duplicate times collapse to the last authored pose.
struct CCameraSequenceTrackedTargetTrack
{
    struct SKeyframe
    {
        float time = 0.f;
        CCameraSequenceTrackedTargetPose pose = {};
    };

    std::vector<SKeyframe> keyframes;
};

/// @brief Defaults shared by all camera-sequence segments unless overridden locally.
struct CCameraSequenceSegmentDefaults
{
    float durationSeconds = 4.f;
    std::vector<CCameraSequencePresentation> presentations;
    CCameraSequenceContinuitySettings continuity = {};
    std::vector<float> captureFractions = { 1.f };
    bool resetCamera = true;
};

/// @brief Authored reusable camera-sequence segment.
/// A segment is the main unit of authored playback and validation and usually maps to one camera showcase chunk.
struct CCameraSequenceSegment
{
    std::string name;
    ICamera::CameraKind cameraKind = ICamera::CameraKind::Unknown;
    std::string cameraIdentifier;

    bool hasDurationSeconds = false;
    float durationSeconds = 0.f;

    bool hasResetCamera = false;
    bool resetCamera = true;

    std::vector<CCameraSequencePresentation> presentations;

    bool hasContinuity = false;
    CCameraSequenceContinuitySettings continuity = {};

    bool hasCaptureFractions = false;
    std::vector<float> captureFractions;

    std::vector<CCameraSequenceKeyframe> keyframes;
    std::vector<CCameraSequenceTrackedTargetKeyframe> targetKeyframes;
};

/// @brief Top-level reusable camera-sequence script.
///
/// This type stores the compact authored description that is later expanded
/// into runtime playback and check payloads.
struct CCameraSequenceScript
{
    bool enabled = true;
    bool log = false;
    bool exclusive = false;
    bool hardFail = false;
    bool visualDebug = false;
    float visualDebugTargetFps = 0.f;
    float visualDebugHoldSeconds = 0.f;
    bool hasEnableActiveCameraMovement = false;
    bool enableActiveCameraMovement = true;
    std::string capturePrefix = "script";
    float fps = 60.f;
    CCameraSequenceSegmentDefaults defaults = {};
    std::vector<CCameraSequenceSegment> segments;
};

/// @brief Reusable compiled sequence segment derived from authored data plus captured references.
/// Consumers can build their own runtime actions/checks from this normalized representation.
struct CCameraSequenceCompiledSegment
{
    std::string name;
    std::vector<CCameraSequencePresentation> presentations;
    CCameraSequenceContinuitySettings continuity = {};
    bool resetCamera = true;
    float durationSeconds = 0.f;
    uint64_t durationFrames = 0ull;
    std::vector<float> sampleTimes;
    std::vector<uint64_t> captureFrameOffsets;
    CCameraKeyframeTrack track = {};
    CCameraSequenceTrackedTargetTrack trackedTargetTrack = {};

    inline bool usesTrackedTargetTrack() const
    {
        return !trackedTargetTrack.keyframes.empty();
    }
};

/// @brief One compiled frame policy entry derived from a reusable compiled segment.
/// Consumers can map these booleans to their own runtime checks and capture requests.
struct CCameraSequenceCompiledFramePolicy
{
    uint64_t frameOffset = 0ull;
    float sampleTime = 0.f;
    bool capture = false;
    bool baseline = false;
    bool continuityStep = false;
    bool followTargetLock = false;
};

struct CCameraSequenceScriptUtilities final
{
    static bool tryParseCameraKind(std::string_view value, ICamera::CameraKind& outKind);
    static bool tryParseProjectionType(std::string_view value, IPlanarProjection::CProjection::ProjectionType& outType);
    static void normalizeCaptureFractions(std::vector<float>& fractions);
    static bool buildSequenceKeyframePreset(const CCameraPreset& reference, const CCameraSequenceKeyframe& authored, CCameraPreset& outPreset, std::string* error = nullptr);
    static bool buildSequenceTrackFromReference(const CCameraPreset& reference, const CCameraSequenceSegment& segment, CCameraKeyframeTrack& outTrack, std::string* error = nullptr);
    static bool isSequenceTrackedTargetPoseFinite(const CCameraSequenceTrackedTargetPose& pose);
    static bool buildSequenceTrackedTargetPoseFromReference(
        const CCameraSequenceTrackedTargetPose& reference,
        const CCameraSequenceTrackedTargetKeyframe& authored,
        CCameraSequenceTrackedTargetPose& outPose,
        std::string* error = nullptr);
    static bool buildSequenceTrackedTargetTrackFromReference(
        const CCameraSequenceTrackedTargetPose& reference,
        const CCameraSequenceSegment& segment,
        CCameraSequenceTrackedTargetTrack& outTrack,
        std::string* error = nullptr);
    static bool tryBuildSequenceTrackedTargetPoseAtTime(
        const CCameraSequenceTrackedTargetTrack& track,
        float time,
        CCameraSequenceTrackedTargetPose& outPose);
    static bool sequenceSegmentUsesTrackedTargetTrack(const CCameraSequenceSegment& segment);
    static float getSequenceSegmentDurationSeconds(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment, const CCameraKeyframeTrack* track = nullptr);
    static const std::vector<CCameraSequencePresentation>& getSequenceSegmentPresentations(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment);
    static CCameraSequenceContinuitySettings getSequenceSegmentContinuity(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment);
    static std::vector<float> getSequenceSegmentCaptureFractions(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment);
    static bool getSequenceSegmentResetCamera(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment);
    static bool sequenceScriptUsesMultiplePresentations(const CCameraSequenceScript& script);
    static uint64_t buildSequenceDurationFrames(float durationSeconds, float fps);
    static void buildSequenceSampleTimes(float durationSeconds, uint64_t durationFrames, std::vector<float>& outTimes);
    static void buildSequenceCaptureFrameOffsets(
        uint64_t durationFrames,
        const std::vector<float>& captureFractions,
        std::vector<uint64_t>& outOffsets);
    static bool compileSequenceSegmentFromReference(
        const CCameraSequenceScript& script,
        const CCameraSequenceSegment& segment,
        const CCameraPreset& referencePreset,
        const CCameraSequenceTrackedTargetPose& referenceTrackedTargetPose,
        CCameraSequenceCompiledSegment& outSegment,
        std::string* error = nullptr);
    static bool buildCompiledSegmentFramePolicies(
        const CCameraSequenceCompiledSegment& segment,
        std::vector<CCameraSequenceCompiledFramePolicy>& outPolicies,
        bool includeFollowTargetLock = false);
};

} // namespace nbl::core

#endif // _C_CAMERA_SEQUENCE_SCRIPT_HPP_

