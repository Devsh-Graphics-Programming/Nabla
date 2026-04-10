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
    static inline bool tryParseCameraKind(std::string_view value, ICamera::CameraKind& outKind)
    {
        if (value == "FPS")
            outKind = ICamera::CameraKind::FPS;
        else if (value == "Free")
            outKind = ICamera::CameraKind::Free;
        else if (value == "Orbit")
            outKind = ICamera::CameraKind::Orbit;
        else if (value == "Arcball")
            outKind = ICamera::CameraKind::Arcball;
        else if (value == "Turntable")
            outKind = ICamera::CameraKind::Turntable;
        else if (value == "TopDown")
            outKind = ICamera::CameraKind::TopDown;
        else if (value == "Isometric")
            outKind = ICamera::CameraKind::Isometric;
        else if (value == "Chase")
            outKind = ICamera::CameraKind::Chase;
        else if (value == "Dolly")
            outKind = ICamera::CameraKind::Dolly;
        else if (value == "DollyZoom" || value == "Dolly Zoom")
            outKind = ICamera::CameraKind::DollyZoom;
        else if (value == "PathRig" || value == "Path Rig")
            outKind = ICamera::CameraKind::Path;
        else
            return false;

        return true;
    }

    static inline bool tryParseProjectionType(std::string_view value, IPlanarProjection::CProjection::ProjectionType& outType)
    {
        if (value == "perspective" || value == "Perspective")
            outType = IPlanarProjection::CProjection::Perspective;
        else if (value == "orthographic" || value == "Orthographic")
            outType = IPlanarProjection::CProjection::Orthographic;
        else
            return false;

        return true;
    }

    static inline void normalizeCaptureFractions(std::vector<float>& fractions)
    {
        for (auto& fraction : fractions)
            fraction = std::clamp(fraction, 0.f, 1.f);

        std::sort(fractions.begin(), fractions.end());
        fractions.erase(std::unique(fractions.begin(), fractions.end(),
            [](const float lhs, const float rhs) { return hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs, rhs, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)); }),
            fractions.end());
    }

    static inline bool buildSequenceKeyframePreset(const CCameraPreset& reference, const CCameraSequenceKeyframe& authored, CCameraPreset& outPreset, std::string* error = nullptr)
    {
        if (authored.hasAbsolutePreset)
        {
            outPreset = authored.absolutePreset;
            if (outPreset.identifier.empty())
                outPreset.identifier = reference.identifier;
            if (outPreset.name.empty())
                outPreset.name = reference.name;
            return CCameraGoalUtilities::isGoalFinite(CCameraPresetUtilities::makeGoalFromPreset(outPreset));
        }

        outPreset = reference;
        if (!authored.hasDelta)
            return true;

        auto goal = CCameraPresetUtilities::makeGoalFromPreset(reference);
        const auto& delta = authored.delta;

        const bool hasPoseDelta = delta.hasPositionOffset || delta.hasRotationEulerDegOffset;
        const bool hasSphericalDelta = delta.hasTargetOffset || delta.orbitDelta.hasAny();
        const bool hasPathDelta = delta.pathDelta.hasAny();

        if (hasPoseDelta && (hasSphericalDelta || hasPathDelta))
        {
            if (error)
                *error = "Sequence keyframe delta cannot mix pose offsets with spherical/path deltas.";
            return false;
        }

        if (delta.hasPositionOffset)
            goal.position += delta.positionOffset;

        if (delta.hasRotationEulerDegOffset)
        {
            goal.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(goal.orientation * hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(delta.rotationEulerDegOffset)));
        }

        if (delta.hasTargetOffset)
        {
            if (!goal.hasTargetPosition)
            {
                if (error)
                    *error = "Sequence keyframe target_offset requires target state.";
                return false;
            }
            goal.targetPosition += delta.targetOffset;
        }

        if (delta.orbitDelta.hasAny())
        {
            if (!goal.hasOrbitState)
            {
                if (error)
                    *error = "Sequence keyframe orbit deltas require spherical orbit state.";
                return false;
            }

            if (delta.orbitDelta.hasU)
                goal.orbitUv.x = hlsl::CCameraMathUtilities::wrapAngleRad(goal.orbitUv.x + delta.orbitDelta.uvDeltaRad.x);
            if (delta.orbitDelta.hasV)
            {
                goal.orbitUv.y = std::clamp(
                    goal.orbitUv.y + delta.orbitDelta.uvDeltaRad.y,
                    -SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad,
                    SCameraTargetRelativeRigDefaults::ArcballPitchLimitRad);
            }
            if (delta.orbitDelta.hasDistance)
                goal.orbitDistance += delta.orbitDelta.distanceDelta;
        }

        if (delta.pathDelta.hasAny())
        {
            if (!goal.hasPathState)
            {
                if (error)
                    *error = "Sequence keyframe path deltas require path state.";
                return false;
            }

            if (!CCameraPathUtilities::tryApplyPathStateDelta(
                    goal.pathState,
                    delta.pathDelta.buildAppliedDelta(),
                    CCameraPathUtilities::makeDefaultPathLimits(),
                    goal.pathState))
            {
                if (error)
                    *error = "Sequence keyframe path deltas produced an invalid path state.";
                return false;
            }
        }

        if (delta.hasDynamicBaseFovDelta || delta.hasDynamicReferenceDistanceDelta)
        {
            if (!goal.hasDynamicPerspectiveState)
            {
                if (error)
                    *error = "Sequence keyframe dynamic perspective deltas require dynamic perspective state.";
                return false;
            }
            if (delta.hasDynamicBaseFovDelta)
                goal.dynamicPerspectiveState.baseFov = std::clamp(goal.dynamicPerspectiveState.baseFov + delta.dynamicBaseFovDelta, 1.f, 179.f);
            if (delta.hasDynamicReferenceDistanceDelta)
                goal.dynamicPerspectiveState.referenceDistance = std::max(0.001f, goal.dynamicPerspectiveState.referenceDistance + delta.dynamicReferenceDistanceDelta);
        }

        if (hasPathDelta || hasSphericalDelta)
        {
            if (!CCameraGoalUtilities::applyCanonicalGoalState(goal))
            {
                if (error)
                    *error = hasPathDelta ?
                        "Sequence keyframe failed to canonicalize path state." :
                        "Sequence keyframe failed to canonicalize spherical state.";
                return false;
            }
        }

        if (!CCameraGoalUtilities::isGoalFinite(goal))
        {
            if (error)
                *error = "Sequence keyframe produced a non-finite goal.";
            return false;
        }

        CCameraPresetUtilities::assignGoalToPreset(outPreset, goal);
        return true;
    }

    static inline bool buildSequenceTrackFromReference(const CCameraPreset& reference, const CCameraSequenceSegment& segment, CCameraKeyframeTrack& outTrack, std::string* error = nullptr)
    {
        outTrack = {};
        outTrack.keyframes.reserve(segment.keyframes.size());

        for (const auto& entry : segment.keyframes)
        {
            CCameraKeyframe keyframe;
            keyframe.time = std::max(0.f, entry.time);
            if (!buildSequenceKeyframePreset(reference, entry, keyframe.preset, error))
                return false;
            outTrack.keyframes.emplace_back(std::move(keyframe));
        }

        CCameraKeyframeTrackUtilities::sortKeyframeTrackByTime(outTrack);
        CCameraKeyframeTrackUtilities::normalizeSelectedKeyframeTrack(outTrack);
        return !outTrack.keyframes.empty();
    }

    static inline bool isSequenceTrackedTargetPoseFinite(const CCameraSequenceTrackedTargetPose& pose)
    {
        return hlsl::CCameraMathUtilities::isFiniteVec3(pose.position) &&
            hlsl::CCameraMathUtilities::isFiniteQuaternion(pose.orientation);
    }

    static inline bool buildSequenceTrackedTargetPoseFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceTrackedTargetKeyframe& authored,
    CCameraSequenceTrackedTargetPose& outPose,
    std::string* error = nullptr)
    {
        outPose = reference;

    if (authored.hasAbsolutePosition)
        outPose.position = authored.absolutePosition;
    if (authored.hasAbsoluteRotationEulerDeg)
        outPose.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(authored.absoluteRotationEulerDeg));

    if (authored.hasDelta)
    {
        if (authored.delta.hasPositionOffset)
            outPose.position += authored.delta.positionOffset;
        if (authored.delta.hasRotationEulerDegOffset)
            outPose.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(outPose.orientation * hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(authored.delta.rotationEulerDegOffset)));
    }

    if (!isSequenceTrackedTargetPoseFinite(outPose))
    {
        if (error)
            *error = "Sequence target keyframe produced a non-finite pose.";
        return false;
    }

        return true;
    }

    static inline bool buildSequenceTrackedTargetTrackFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceSegment& segment,
    CCameraSequenceTrackedTargetTrack& outTrack,
    std::string* error = nullptr)
    {
        outTrack = {};
        outTrack.keyframes.reserve(segment.targetKeyframes.size());

    for (const auto& entry : segment.targetKeyframes)
    {
        CCameraSequenceTrackedTargetTrack::SKeyframe keyframe;
        keyframe.time = std::max(0.f, entry.time);
        if (!buildSequenceTrackedTargetPoseFromReference(reference, entry, keyframe.pose, error))
            return false;
        outTrack.keyframes.emplace_back(std::move(keyframe));
    }

    std::stable_sort(outTrack.keyframes.begin(), outTrack.keyframes.end(),
        [](const auto& lhs, const auto& rhs)
        {
            if (lhs.time == rhs.time)
                return false;
            return lhs.time < rhs.time;
        });

    std::vector<CCameraSequenceTrackedTargetTrack::SKeyframe> normalized;
    normalized.reserve(outTrack.keyframes.size());
    for (const auto& keyframe : outTrack.keyframes)
    {
        if (!normalized.empty() && hlsl::CCameraMathUtilities::nearlyEqualScalar(normalized.back().time, keyframe.time, static_cast<float>(SCameraToolingThresholds::ScalarTolerance)))
            normalized.back() = keyframe;
        else
            normalized.emplace_back(keyframe);
    }
    outTrack.keyframes = std::move(normalized);

        return !outTrack.keyframes.empty();
    }

    static inline bool tryBuildSequenceTrackedTargetPoseAtTime(
    const CCameraSequenceTrackedTargetTrack& track,
    const float time,
    CCameraSequenceTrackedTargetPose& outPose)
    {
        if (track.keyframes.empty())
            return false;
        if (track.keyframes.size() == 1u || time <= track.keyframes.front().time)
        {
            outPose = track.keyframes.front().pose;
            return true;
        }
        if (time >= track.keyframes.back().time)
        {
            outPose = track.keyframes.back().pose;
            return true;
        }

        for (size_t ix = 1u; ix < track.keyframes.size(); ++ix)
        {
            const auto& lhs = track.keyframes[ix - 1u];
            const auto& rhs = track.keyframes[ix];
            if (time > rhs.time)
                continue;

            const auto span = std::max(static_cast<float>(SCameraToolingThresholds::ScalarTolerance), rhs.time - lhs.time);
            const auto alpha = std::clamp((time - lhs.time) / span, 0.f, 1.f);
            outPose.position = lhs.pose.position + (rhs.pose.position - lhs.pose.position) * static_cast<double>(alpha);
            outPose.orientation = hlsl::CCameraMathUtilities::slerpQuaternion(lhs.pose.orientation, rhs.pose.orientation, static_cast<hlsl::float64_t>(alpha));
            return true;
        }

        outPose = track.keyframes.back().pose;
        return true;
    }
    
    static inline bool sequenceSegmentUsesTrackedTargetTrack(const CCameraSequenceSegment& segment)
    {
        return !segment.targetKeyframes.empty();
    }

    static inline float getSequenceSegmentDurationSeconds(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment, const CCameraKeyframeTrack* track = nullptr)
    {
        if (segment.hasDurationSeconds)
            return std::max(0.f, segment.durationSeconds);
        if (script.defaults.durationSeconds > 0.f)
            return script.defaults.durationSeconds;
        if (track)
            return track->keyframes.empty() ? 0.f : track->keyframes.back().time;
        return 0.f;
    }

    static inline const std::vector<CCameraSequencePresentation>& getSequenceSegmentPresentations(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
    {
        return segment.presentations.empty() ? script.defaults.presentations : segment.presentations;
    }

    static inline CCameraSequenceContinuitySettings getSequenceSegmentContinuity(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
    {
        return segment.hasContinuity ? segment.continuity : script.defaults.continuity;
    }

    static inline std::vector<float> getSequenceSegmentCaptureFractions(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
    {
        auto captures = segment.hasCaptureFractions ? segment.captureFractions : script.defaults.captureFractions;
        normalizeCaptureFractions(captures);
        return captures;
    }

    static inline bool getSequenceSegmentResetCamera(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
    {
        return segment.hasResetCamera ? segment.resetCamera : script.defaults.resetCamera;
    }

    static inline bool sequenceScriptUsesMultiplePresentations(const CCameraSequenceScript& script)
    {
        if (script.defaults.presentations.size() > 1u)
            return true;

        for (const auto& segment : script.segments)
        {
            if (getSequenceSegmentPresentations(script, segment).size() > 1u)
                return true;
        }

        return false;
    }

    static inline uint64_t buildSequenceDurationFrames(const float durationSeconds, const float fps)
    {
        const auto safeDuration = std::max(0.f, durationSeconds);
        const auto safeFps = std::max(1.f, fps);
        return std::max<uint64_t>(1ull, static_cast<uint64_t>(std::llround(static_cast<double>(safeDuration) * static_cast<double>(safeFps))));
    }

    /// @brief Build one sampled time per authored frame in the compiled segment.
    static inline void buildSequenceSampleTimes(const float durationSeconds, const uint64_t durationFrames, std::vector<float>& outTimes)
    {
        outTimes.clear();
        outTimes.reserve(durationFrames);

        for (uint64_t frameOffset = 0u; frameOffset < durationFrames; ++frameOffset)
        {
            const float alpha = durationFrames > 1u ? static_cast<float>(frameOffset) / static_cast<float>(durationFrames - 1u) : 0.f;
            outTimes.emplace_back(durationSeconds * alpha);
        }
    }

    /// @brief Expand normalized capture fractions into concrete frame offsets inside the compiled segment.
    static inline void buildSequenceCaptureFrameOffsets(
    const uint64_t durationFrames,
    const std::vector<float>& captureFractions,
    std::vector<uint64_t>& outOffsets)
    {
        outOffsets.clear();
        outOffsets.reserve(captureFractions.size());

    for (const auto fraction : captureFractions)
    {
        const auto offset = durationFrames > 1u ?
            static_cast<uint64_t>(std::llround(static_cast<double>(fraction) * static_cast<double>(durationFrames - 1u))) :
            0ull;
        outOffsets.emplace_back(offset);
    }

    std::sort(outOffsets.begin(), outOffsets.end());
        outOffsets.erase(std::unique(outOffsets.begin(), outOffsets.end()), outOffsets.end());
    }

    /// @brief Compile one authored sequence segment into normalized reusable data for runtime consumers.
    static inline bool compileSequenceSegmentFromReference(
    const CCameraSequenceScript& script,
    const CCameraSequenceSegment& segment,
    const CCameraPreset& referencePreset,
    const CCameraSequenceTrackedTargetPose& referenceTrackedTargetPose,
    CCameraSequenceCompiledSegment& outSegment,
    std::string* error = nullptr)
    {
        outSegment = {};
        outSegment.name = segment.name;
        outSegment.presentations = getSequenceSegmentPresentations(script, segment);
        outSegment.continuity = getSequenceSegmentContinuity(script, segment);
        outSegment.resetCamera = getSequenceSegmentResetCamera(script, segment);

        if (!buildSequenceTrackFromReference(referencePreset, segment, outSegment.track, error))
            return false;

        if (sequenceSegmentUsesTrackedTargetTrack(segment) &&
            !buildSequenceTrackedTargetTrackFromReference(referenceTrackedTargetPose, segment, outSegment.trackedTargetTrack, error))
        {
            return false;
        }

        outSegment.durationSeconds = getSequenceSegmentDurationSeconds(script, segment, &outSegment.track);
        outSegment.durationFrames = buildSequenceDurationFrames(outSegment.durationSeconds, script.fps);
        buildSequenceSampleTimes(outSegment.durationSeconds, outSegment.durationFrames, outSegment.sampleTimes);
        buildSequenceCaptureFrameOffsets(outSegment.durationFrames, getSequenceSegmentCaptureFractions(script, segment), outSegment.captureFrameOffsets);
        return true;
    }

    static inline bool buildCompiledSegmentFramePolicies(
    const CCameraSequenceCompiledSegment& segment,
    std::vector<CCameraSequenceCompiledFramePolicy>& outPolicies,
    const bool includeFollowTargetLock = false)
    {
        if (segment.sampleTimes.size() != segment.durationFrames)
            return false;

    outPolicies.clear();
    outPolicies.reserve(segment.durationFrames);

    size_t captureIx = 0u;
    for (uint64_t frameOffset = 0u; frameOffset < segment.durationFrames; ++frameOffset)
    {
        CCameraSequenceCompiledFramePolicy policy;
        policy.frameOffset = frameOffset;
        policy.sampleTime = segment.sampleTimes[frameOffset];
        policy.baseline = segment.continuity.baseline && frameOffset == 0u;
        policy.continuityStep = segment.continuity.step && frameOffset > 0u;
        policy.followTargetLock = includeFollowTargetLock && segment.usesTrackedTargetTrack() && policy.continuityStep;

        while (captureIx < segment.captureFrameOffsets.size() && segment.captureFrameOffsets[captureIx] < frameOffset)
            ++captureIx;
        policy.capture = captureIx < segment.captureFrameOffsets.size() && segment.captureFrameOffsets[captureIx] == frameOffset;
        if (policy.capture)
            ++captureIx;

        outPolicies.emplace_back(std::move(policy));
    }

        return true;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_SEQUENCE_SCRIPT_HPP_

