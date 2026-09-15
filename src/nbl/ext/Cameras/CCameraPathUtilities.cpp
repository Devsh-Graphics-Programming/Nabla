// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraPathUtilities.hpp"

namespace nbl::core
{

ICamera::PathState CCameraPathUtilities::makeDefaultPathState(const double minU)
{
    return {
        .s = 0.0,
        .u = minU,
        .v = 0.0,
        .roll = 0.0
    };
}

SCameraPathComparisonThresholds CCameraPathUtilities::makePathComparisonThresholds(
    const double angularToleranceDeg,
    const double scalarTolerance)
{
    return {
        .sToleranceDeg = angularToleranceDeg,
        .rollToleranceDeg = angularToleranceDeg,
        .scalarTolerance = scalarTolerance
    };
}

bool CCameraPathUtilities::isPathStateFinite(const ICamera::PathState& state)
{
    return hlsl::CCameraMathUtilities::isFiniteScalar(state.s) &&
        hlsl::CCameraMathUtilities::isFiniteScalar(state.u) &&
        hlsl::CCameraMathUtilities::isFiniteScalar(state.v) &&
        hlsl::CCameraMathUtilities::isFiniteScalar(state.roll);
}

bool CCameraPathUtilities::isPathLimitsWellFormed(const SCameraPathLimits& limits)
{
    return hlsl::CCameraMathUtilities::isFiniteScalar(limits.minU) &&
        hlsl::CCameraMathUtilities::isFiniteScalar(limits.minDistance) &&
        !std::isnan(static_cast<double>(limits.maxDistance));
}

bool CCameraPathUtilities::sanitizePathLimits(SCameraPathLimits& limits)
{
    if (!isPathLimitsWellFormed(limits))
        return false;

    limits.minU = std::max(limits.minU, 0.0);
    limits.minDistance = std::max<hlsl::float64_t>(
        std::max<hlsl::float64_t>(limits.minDistance, static_cast<hlsl::float64_t>(limits.minU)),
        static_cast<hlsl::float64_t>(SCameraTargetRelativeTraits::MinDistance));

    if (!std::isfinite(static_cast<double>(limits.maxDistance)))
        limits.maxDistance = std::numeric_limits<hlsl::float64_t>::infinity();
    else
        limits.maxDistance = std::max(limits.maxDistance, limits.minDistance);
    return true;
}

bool CCameraPathUtilities::sanitizePathState(ICamera::PathState& state, const double minU)
{
    return hlsl::CCameraMathUtilities::sanitizePathState(state.s, state.u, state.v, state.roll, minU);
}

bool CCameraPathUtilities::sanitizePathState(ICamera::PathState& state, const SCameraPathLimits& limits, double* outAppliedDistance)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits))
        return false;

    if (!sanitizePathState(state, sanitizedLimits.minU))
        return false;

    const auto desiredDistance = std::clamp(
        hlsl::CCameraMathUtilities::getPathDistance(state.u, state.v),
        sanitizedLimits.minDistance,
        sanitizedLimits.maxDistance);
    return tryScalePathStateDistance(desiredDistance, sanitizedLimits.minU, state, outAppliedDistance);
}

bool CCameraPathUtilities::tryScalePathStateDistance(
    const double desiredDistance,
    const double minU,
    ICamera::PathState& ioState,
    double* outAppliedDistance)
{
    return hlsl::CCameraMathUtilities::tryScalePathStateDistance(
        desiredDistance,
        minU,
        ioState.u,
        ioState.v,
        outAppliedDistance);
}

bool CCameraPathUtilities::tryUpdatePathStateDistance(
    const float desiredDistance,
    const SCameraPathLimits& limits,
    ICamera::PathState& ioState,
    SCameraPathDistanceUpdateResult* outResult)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits) || !sanitizePathState(ioState, sanitizedLimits))
        return false;

    const auto clampedDistance = std::clamp<hlsl::float64_t>(desiredDistance, sanitizedLimits.minDistance, sanitizedLimits.maxDistance);
    double appliedDistance = 0.0;
    if (!tryScalePathStateDistance(static_cast<double>(clampedDistance), sanitizedLimits.minU, ioState, &appliedDistance))
        return false;

    if (outResult)
    {
        outResult->appliedDistance = appliedDistance;
        outResult->exact = (clampedDistance == desiredDistance) &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(appliedDistance, static_cast<double>(desiredDistance), SCameraPathDefaults::ScalarTolerance);
    }
    return true;
}

bool CCameraPathUtilities::tryBuildPathStateFromPosition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const double minU,
    ICamera::PathState& outState)
{
    outState = {};
    if (!hlsl::CCameraMathUtilities::tryBuildPathStateFromPosition(
            targetPosition,
            position,
            minU,
            outState.s,
            outState.u,
            outState.v))
    {
        return false;
    }

    outState.roll = 0.0;
    return true;
}

bool CCameraPathUtilities::tryResolvePathState(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const SCameraPathLimits& limits,
    const ICamera::PathState* requestedState,
    ICamera::PathState& outState)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits))
        return false;

    if (requestedState)
    {
        outState = *requestedState;
        return sanitizePathState(outState, sanitizedLimits);
    }

    if (tryBuildPathStateFromPosition(targetPosition, position, sanitizedLimits.minU, outState))
        return sanitizePathState(outState, sanitizedLimits);

    outState = makeDefaultPathState(sanitizedLimits.minU);
    return sanitizePathState(outState, sanitizedLimits);
}

bool CCameraPathUtilities::tryBuildPathPoseFromState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    SCameraPathPose& outPose)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits))
        return false;

    return hlsl::CCameraMathUtilities::tryBuildPathPoseFromState(
        targetPosition,
        state.s,
        state.u,
        state.v,
        state.roll,
        sanitizedLimits.minU,
        sanitizedLimits.minDistance,
        sanitizedLimits.maxDistance,
        outPose.position,
        outPose.orientation,
        &outPose.appliedDistance,
        &outPose.orbitUv);
}

bool CCameraPathUtilities::tryBuildPathPoseFromState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    hlsl::float64_t3& outPosition,
    hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation,
    hlsl::float64_t* outAppliedDistance,
    hlsl::float64_t2* outOrbitUv)
{
    SCameraPathPose pathPose = {};
    if (!tryBuildPathPoseFromState(targetPosition, state, limits, pathPose))
        return false;

    outPosition = pathPose.position;
    outOrientation = pathPose.orientation;
    if (outAppliedDistance)
        *outAppliedDistance = pathPose.appliedDistance;
    if (outOrbitUv)
        *outOrbitUv = pathPose.orbitUv;
    return true;
}

bool CCameraPathUtilities::pathStatesNearlyEqual(
    const ICamera::PathState& lhs,
    const ICamera::PathState& rhs,
    const SCameraPathComparisonThresholds& thresholds)
{
    return hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.s, rhs.s) <= thresholds.sToleranceDeg &&
        hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.u, rhs.u, thresholds.scalarTolerance) &&
        hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.v, rhs.v, thresholds.scalarTolerance) &&
        hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.roll, rhs.roll) <= thresholds.rollToleranceDeg;
}

bool CCameraPathUtilities::pathStatesChanged(
    const ICamera::PathState& lhs,
    const ICamera::PathState& rhs,
    const SCameraPathComparisonThresholds& thresholds)
{
    return !pathStatesNearlyEqual(lhs, rhs, thresholds);
}

hlsl::float64_t4 CCameraPathUtilities::buildPathStateDeltaVector(
    const ICamera::PathState& currentState,
    const ICamera::PathState& desiredState)
{
    auto deltaVector = desiredState.asVector() - currentState.asVector();
    deltaVector.x = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.x);
    deltaVector.w = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.w);
    return deltaVector;
}

SCameraPathDelta CCameraPathUtilities::buildPathStateDelta(
    const ICamera::PathState& currentState,
    const ICamera::PathState& desiredState)
{
    return SCameraPathDelta::fromVector(buildPathStateDeltaVector(currentState, desiredState));
}

SCameraPathDelta CCameraPathUtilities::makePathDeltaFromVirtualPathMotion(
    const hlsl::float64_t3& translation,
    const hlsl::float64_t3& rotation)
{
    return SCameraPathDelta::fromMotion(translation, rotation.z);
}

SCameraPathDelta CCameraPathUtilities::buildDefaultPathControlDelta(const SCameraPathControlContext& context)
{
    return makePathDeltaFromVirtualPathMotion(context.translation, context.rotation);
}

void CCameraPathUtilities::appendPathDeltaEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const SCameraPathDelta& delta,
    const double moveDenominator,
    const double rotationDenominator,
    const SCameraPathComparisonThresholds& thresholds)
{
    CCameraVirtualEventUtilities::appendLocalTranslationEvents(
        events,
        delta.translationVector(),
        hlsl::float64_t3(moveDenominator),
        hlsl::float64_t3(thresholds.scalarTolerance));
    CCameraVirtualEventUtilities::appendAngularDeltaEvent(
        events,
        delta.roll,
        rotationDenominator,
        thresholds.rollToleranceDeg,
        CVirtualGimbalEvent::RollRight,
        CVirtualGimbalEvent::RollLeft);
}

bool CCameraPathUtilities::tryBuildCanonicalPathState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    SCameraCanonicalPathState& outState)
{
    outState = {};
    if (!tryBuildPathPoseFromState(targetPosition, state, limits, outState.pose))
        return false;

    outState.targetRelative = {
        .target = targetPosition,
        .orbitUv = outState.pose.orbitUv,
        .distance = static_cast<float>(outState.pose.appliedDistance)
    };
    return true;
}

bool CCameraPathUtilities::tryApplyPathStateDelta(
    const ICamera::PathState& currentState,
    const SCameraPathDelta& delta,
    const SCameraPathLimits& limits,
    ICamera::PathState& outState)
{
    auto stateVector = currentState.asVector() + delta.asVector();
    stateVector.x = hlsl::CCameraMathUtilities::wrapAngleRad(stateVector.x);
    stateVector.w = hlsl::CCameraMathUtilities::wrapAngleRad(stateVector.w);
    outState = ICamera::PathState::fromVector(stateVector);
    return sanitizePathState(outState, limits);
}

ICamera::PathState CCameraPathUtilities::blendPathStates(
    const ICamera::PathState& from,
    const ICamera::PathState& to,
    const double alpha)
{
    const auto fromVector = from.asVector();
    const auto toVector = to.asVector();
    return {
        .s = hlsl::CCameraMathUtilities::lerpWrappedAngleRad(fromVector.x, toVector.x, alpha),
        .u = fromVector.y + (toVector.y - fromVector.y) * alpha,
        .v = fromVector.z + (toVector.z - fromVector.z) * alpha,
        .roll = hlsl::CCameraMathUtilities::lerpWrappedAngleRad(fromVector.w, toVector.w, alpha)
    };
}

bool CCameraPathUtilities::tryBuildPathStateTransition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& currentPosition,
    const hlsl::float64_t3& desiredPosition,
    const SCameraPathLimits& limits,
    const ICamera::PathState* currentStateOverride,
    const ICamera::PathState* desiredStateOverride,
    SCameraPathStateTransition& outTransition)
{
    if (!tryResolvePathState(targetPosition, currentPosition, limits, currentStateOverride, outTransition.current))
        return false;
    if (!tryResolvePathState(targetPosition, desiredPosition, limits, desiredStateOverride, outTransition.desired))
        return false;

    outTransition.delta = buildPathStateDelta(outTransition.current, outTransition.desired);
    return true;
}

SCameraPathModel CCameraPathUtilities::makeDefaultPathModel()
{
    return {
        .resolveState =
            [](const hlsl::float64_t3& targetPosition,
                const hlsl::float64_t3& position,
                const SCameraPathLimits& limits,
                const ICamera::PathState* requestedState,
                ICamera::PathState& outState) -> bool
            {
                return tryResolvePathState(targetPosition, position, limits, requestedState, outState);
            },
        .controlLaw =
            [](const SCameraPathControlContext& context) -> SCameraPathDelta
            {
                return buildDefaultPathControlDelta(context);
            },
        .integrate =
            [](const ICamera::PathState& currentState,
                const SCameraPathDelta& delta,
                const SCameraPathLimits& limits,
                ICamera::PathState& outState) -> bool
            {
                return tryApplyPathStateDelta(currentState, delta, limits, outState);
            },
        .evaluate =
            [](const hlsl::float64_t3& targetPosition,
                const ICamera::PathState& state,
                const SCameraPathLimits& limits,
                SCameraCanonicalPathState& outState) -> bool
            {
                return tryBuildCanonicalPathState(targetPosition, state, limits, outState);
            },
        .updateDistance =
            [](const float desiredDistance,
                const SCameraPathLimits& limits,
                ICamera::PathState& ioState,
                SCameraPathDistanceUpdateResult* outResult) -> bool
            {
                return tryUpdatePathStateDistance(desiredDistance, limits, ioState, outResult);
            }
    };
}

} // namespace nbl::core
