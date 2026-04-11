// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraSequenceScript.hpp"

namespace nbl::core
{

bool CCameraSequenceScriptUtilities::tryParseCameraKind(std::string_view value, ICamera::CameraKind& outKind)
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

bool CCameraSequenceScriptUtilities::tryParseProjectionType(std::string_view value, IPlanarProjection::CProjection::ProjectionType& outType)
{
    if (value == "perspective" || value == "Perspective")
        outType = IPlanarProjection::CProjection::Perspective;
    else if (value == "orthographic" || value == "Orthographic")
        outType = IPlanarProjection::CProjection::Orthographic;
    else
        return false;

    return true;
}

void CCameraSequenceScriptUtilities::normalizeCaptureFractions(std::vector<float>& fractions)
{
    for (auto& fraction : fractions)
        fraction = std::clamp(fraction, 0.f, 1.f);

    std::sort(fractions.begin(), fractions.end());
    fractions.erase(
        std::unique(
            fractions.begin(),
            fractions.end(),
            [](const float lhs, const float rhs)
            {
                return hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs, rhs, static_cast<float>(SCameraToolingThresholds::ScalarTolerance));
            }),
        fractions.end());
}

bool CCameraSequenceScriptUtilities::buildSequenceKeyframePreset(const CCameraPreset& reference, const CCameraSequenceKeyframe& authored, CCameraPreset& outPreset, std::string* error)
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
        goal.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(
            goal.orientation * hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
                hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(delta.rotationEulerDegOffset)));
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
        {
            goal.dynamicPerspectiveState.referenceDistance = std::max(
                0.001f,
                goal.dynamicPerspectiveState.referenceDistance + delta.dynamicReferenceDistanceDelta);
        }
    }

    if (hasPathDelta || hasSphericalDelta)
    {
        if (!CCameraGoalUtilities::applyCanonicalGoalState(goal))
        {
            if (error)
            {
                *error = hasPathDelta ?
                    "Sequence keyframe failed to canonicalize path state." :
                    "Sequence keyframe failed to canonicalize spherical state.";
            }
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

bool CCameraSequenceScriptUtilities::buildSequenceTrackFromReference(const CCameraPreset& reference, const CCameraSequenceSegment& segment, CCameraKeyframeTrack& outTrack, std::string* error)
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

bool CCameraSequenceScriptUtilities::isSequenceTrackedTargetPoseFinite(const CCameraSequenceTrackedTargetPose& pose)
{
    return hlsl::CCameraMathUtilities::isFiniteVec3(pose.position) &&
        hlsl::CCameraMathUtilities::isFiniteQuaternion(pose.orientation);
}

bool CCameraSequenceScriptUtilities::buildSequenceTrackedTargetPoseFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceTrackedTargetKeyframe& authored,
    CCameraSequenceTrackedTargetPose& outPose,
    std::string* error)
{
    outPose = reference;

    if (authored.hasAbsolutePosition)
        outPose.position = authored.absolutePosition;
    if (authored.hasAbsoluteRotationEulerDeg)
    {
        outPose.orientation = hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
            hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(authored.absoluteRotationEulerDeg));
    }

    if (authored.hasDelta)
    {
        if (authored.delta.hasPositionOffset)
            outPose.position += authored.delta.positionOffset;
        if (authored.delta.hasRotationEulerDegOffset)
        {
            outPose.orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(
                outPose.orientation * hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
                    hlsl::CCameraMathUtilities::castVector<hlsl::float64_t>(authored.delta.rotationEulerDegOffset)));
        }
    }

    if (!isSequenceTrackedTargetPoseFinite(outPose))
    {
        if (error)
            *error = "Sequence target keyframe produced a non-finite pose.";
        return false;
    }

    return true;
}

bool CCameraSequenceScriptUtilities::buildSequenceTrackedTargetTrackFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceSegment& segment,
    CCameraSequenceTrackedTargetTrack& outTrack,
    std::string* error)
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

    std::stable_sort(
        outTrack.keyframes.begin(),
        outTrack.keyframes.end(),
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
        if (!normalized.empty() &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(
                normalized.back().time,
                keyframe.time,
                static_cast<float>(SCameraToolingThresholds::ScalarTolerance)))
        {
            normalized.back() = keyframe;
        }
        else
        {
            normalized.emplace_back(keyframe);
        }
    }
    outTrack.keyframes = std::move(normalized);

    return !outTrack.keyframes.empty();
}

bool CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(
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

bool CCameraSequenceScriptUtilities::sequenceSegmentUsesTrackedTargetTrack(const CCameraSequenceSegment& segment)
{
    return !segment.targetKeyframes.empty();
}

float CCameraSequenceScriptUtilities::getSequenceSegmentDurationSeconds(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment, const CCameraKeyframeTrack* track)
{
    if (segment.hasDurationSeconds)
        return std::max(0.f, segment.durationSeconds);
    if (script.defaults.durationSeconds > 0.f)
        return script.defaults.durationSeconds;
    if (track)
        return track->keyframes.empty() ? 0.f : track->keyframes.back().time;
    return 0.f;
}

const std::vector<CCameraSequencePresentation>& CCameraSequenceScriptUtilities::getSequenceSegmentPresentations(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.presentations.empty() ? script.defaults.presentations : segment.presentations;
}

CCameraSequenceContinuitySettings CCameraSequenceScriptUtilities::getSequenceSegmentContinuity(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.hasContinuity ? segment.continuity : script.defaults.continuity;
}

std::vector<float> CCameraSequenceScriptUtilities::getSequenceSegmentCaptureFractions(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    auto captures = segment.hasCaptureFractions ? segment.captureFractions : script.defaults.captureFractions;
    normalizeCaptureFractions(captures);
    return captures;
}

bool CCameraSequenceScriptUtilities::getSequenceSegmentResetCamera(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.hasResetCamera ? segment.resetCamera : script.defaults.resetCamera;
}

bool CCameraSequenceScriptUtilities::sequenceScriptUsesMultiplePresentations(const CCameraSequenceScript& script)
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

uint64_t CCameraSequenceScriptUtilities::buildSequenceDurationFrames(const float durationSeconds, const float fps)
{
    const auto safeDuration = std::max(0.f, durationSeconds);
    const auto safeFps = std::max(1.f, fps);
    return std::max<uint64_t>(1ull, static_cast<uint64_t>(std::llround(static_cast<double>(safeDuration) * static_cast<double>(safeFps))));
}

void CCameraSequenceScriptUtilities::buildSequenceSampleTimes(const float durationSeconds, const uint64_t durationFrames, std::vector<float>& outTimes)
{
    outTimes.clear();
    outTimes.reserve(durationFrames);

    for (uint64_t frameOffset = 0u; frameOffset < durationFrames; ++frameOffset)
    {
        const float alpha = durationFrames > 1u ? static_cast<float>(frameOffset) / static_cast<float>(durationFrames - 1u) : 0.f;
        outTimes.emplace_back(durationSeconds * alpha);
    }
}

void CCameraSequenceScriptUtilities::buildSequenceCaptureFrameOffsets(
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

bool CCameraSequenceScriptUtilities::compileSequenceSegmentFromReference(
    const CCameraSequenceScript& script,
    const CCameraSequenceSegment& segment,
    const CCameraPreset& referencePreset,
    const CCameraSequenceTrackedTargetPose& referenceTrackedTargetPose,
    CCameraSequenceCompiledSegment& outSegment,
    std::string* error)
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

bool CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(
    const CCameraSequenceCompiledSegment& segment,
    std::vector<CCameraSequenceCompiledFramePolicy>& outPolicies,
    const bool includeFollowTargetLock)
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

} // namespace nbl::core
