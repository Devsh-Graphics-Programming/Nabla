// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraPathUtilities.hpp"

namespace nbl::ext::cameras
{

namespace
{

hlsl::float64_t getHorizonRadius(const hlsl::float64_t3& offset, const SWorldFrame& frame)
{
    return hlsl::length(offset - frame.up * hlsl::dot(offset, frame.up));
}

hlsl::float64_t getPathDistance(const hlsl::float64_t pathU, const hlsl::float64_t pathV)
{
    return hlsl::length(hlsl::float64_t2(pathU, pathV));
}

// NOTE: `pathS` is measured from the frame's right towards its forward, a quarter turn away from the orbit azimuth
// of `STargetOrbit`; `pathU` is the radius in the horizon plane, `pathV` the height along the frame's up.
hlsl::float64_t3 makePathOffsetFromState(const hlsl::float64_t pathS, const hlsl::float64_t pathU, const hlsl::float64_t pathV, const SWorldFrame& frame)
{
    return frame.right * (hlsl::cos(pathS) * pathU) + frame.up * pathV + frame.forward * (hlsl::sin(pathS) * pathU);
}

}

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
    return CCameraMathUtilities::isFiniteScalar(state.s) &&
        CCameraMathUtilities::isFiniteScalar(state.u) &&
        CCameraMathUtilities::isFiniteScalar(state.v) &&
        CCameraMathUtilities::isFiniteScalar(state.roll);
}

bool CCameraPathUtilities::isPathLimitsWellFormed(const SCameraPathLimits& limits)
{
    return CCameraMathUtilities::isFiniteScalar(limits.minU) &&
        CCameraMathUtilities::isFiniteScalar(limits.minDistance) &&
        !std::isnan(static_cast<double>(limits.maxDistance));
}

bool CCameraPathUtilities::sanitizePathLimits(SCameraPathLimits& limits)
{
    if (!isPathLimitsWellFormed(limits))
        return false;

    limits.minU = std::max(limits.minU, 0.0);
    limits.minDistance = std::max<hlsl::float64_t>(
        std::max<hlsl::float64_t>(limits.minDistance, static_cast<hlsl::float64_t>(limits.minU)),
        static_cast<hlsl::float64_t>(ICamera::DefaultMinTargetDistance));

    if (!std::isfinite(static_cast<double>(limits.maxDistance)))
        limits.maxDistance = std::numeric_limits<hlsl::float64_t>::infinity();
    else
        limits.maxDistance = std::max(limits.maxDistance, limits.minDistance);
    return true;
}

bool CCameraPathUtilities::sanitizePathState(ICamera::PathState& state, const double minU)
{
    if (!isPathStateFinite(state))
        return false;

    state.s = CCameraMathUtilities::wrapAngleRad(state.s);
    state.u = hlsl::max(minU, state.u);
    state.roll = CCameraMathUtilities::wrapAngleRad(state.roll);
    return isPathStateFinite(state);
}

bool CCameraPathUtilities::sanitizePathState(ICamera::PathState& state, const SCameraPathLimits& limits, double* outAppliedDistance)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits))
        return false;

    if (!sanitizePathState(state, sanitizedLimits.minU))
        return false;

    const auto desiredDistance = std::clamp(
        getPathDistance(state.u, state.v),
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
    if (!CCameraMathUtilities::isFiniteScalar(desiredDistance) ||
        !CCameraMathUtilities::isFiniteScalar(ioState.u) ||
        !CCameraMathUtilities::isFiniteScalar(ioState.v))
        return false;

    const hlsl::float64_t currentDistance = getPathDistance(ioState.u, ioState.v);
    constexpr hlsl::float64_t Epsilon = hlsl::numeric_limits<hlsl::float64_t>::epsilon;
    if (currentDistance > Epsilon)
    {
        const hlsl::float64_t scale = desiredDistance / currentDistance;
        ioState.u = hlsl::max(minU, ioState.u * scale);
        ioState.v *= scale;
    }
    else
    {
        ioState.u = hlsl::max(minU, desiredDistance);
        ioState.v = 0.0;
    }

    if (outAppliedDistance)
        *outAppliedDistance = getPathDistance(ioState.u, ioState.v);
    return CCameraMathUtilities::isFiniteScalar(ioState.u) && CCameraMathUtilities::isFiniteScalar(ioState.v);
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
            CCameraMathUtilities::nearlyEqualScalar(appliedDistance, static_cast<double>(desiredDistance), SCameraPathDefaults::ScalarTolerance);
    }
    return true;
}

bool CCameraPathUtilities::tryBuildPathStateFromPosition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const double minU,
    const SWorldFrame& frame,
    ICamera::PathState& outState)
{
    outState = {};
    const auto offset = position - targetPosition;
    const auto radius = getHorizonRadius(offset, frame);
    const auto height = hlsl::dot(offset, frame.up);
    if (!CCameraMathUtilities::isFiniteScalar(radius) || !CCameraMathUtilities::isFiniteScalar(height))
        return false;

    outState.s = CCameraMathUtilities::wrapAngleRad(hlsl::atan2(hlsl::dot(offset, frame.forward), hlsl::dot(offset, frame.right)));
    outState.u = hlsl::max(minU, radius);
    outState.v = height;
    if (!CCameraMathUtilities::isFiniteScalar(outState.s) ||
        !CCameraMathUtilities::isFiniteScalar(outState.u) ||
        !CCameraMathUtilities::isFiniteScalar(outState.v))
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
    const SWorldFrame& frame,
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

    if (tryBuildPathStateFromPosition(targetPosition, position, sanitizedLimits.minU, frame, outState))
        return sanitizePathState(outState, sanitizedLimits);

    outState = makeDefaultPathState(sanitizedLimits.minU);
    return sanitizePathState(outState, sanitizedLimits);
}

bool CCameraPathUtilities::tryBuildPathPoseFromState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    const SWorldFrame& frame,
    SCameraPathPose& outPose)
{
    SCameraPathLimits sanitizedLimits = limits;
    if (!sanitizePathLimits(sanitizedLimits))
        return false;

    if (!isPathStateFinite(state))
        return false;

    const hlsl::float64_t appliedU = hlsl::max(sanitizedLimits.minU, state.u);
    const auto offset = makePathOffsetFromState(state.s, appliedU, state.v, frame);

    STargetOrbit orbit = {};
    if (!CCameraMathUtilities::tryBuildOrbitFromPosition(targetPosition, targetPosition + offset, sanitizedLimits.minDistance, sanitizedLimits.maxDistance, frame, orbit))
        return false;
    SCameraRigPose pose = {};
    if (!CCameraMathUtilities::tryBuildPoseFromOrbit(orbit, sanitizedLimits.minDistance, sanitizedLimits.maxDistance, frame, pose, &orbit.distance))
        return false;

    outPose.position = pose.position;
    outPose.orientation = pose.orientation;
    if (!CCameraMathUtilities::isNearlyZeroScalar(state.roll, hlsl::numeric_limits<hlsl::float64_t>::epsilon))
    {
        // roll turns right and up about forward
        const auto basis = CCameraMathUtilities::basisFromOrientation(outPose.orientation, frame.handedness);
        const hlsl::float64_t rollCos = hlsl::cos(state.roll);
        const hlsl::float64_t rollSin = hlsl::sin(state.roll);
        const auto right = basis.right * rollCos + basis.up * rollSin;
        const auto up = basis.up * rollCos - basis.right * rollSin;
        const auto rolled = CCameraMathUtilities::orientationFromBasis(SCameraBasis<hlsl::float64_t>{ right, up, basis.forward }, frame.handedness);
        if (!CCameraMathUtilities::isFiniteQuaternion(rolled))
            return false;
        outPose.orientation = hlsl::normalize(rolled);
    }

    outPose.appliedDistance = orbit.distance;
    outPose.orbitUv = orbit.angles;
    return true;
}

bool CCameraPathUtilities::tryBuildPathPoseFromState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    const SWorldFrame& frame,
    hlsl::float64_t3& outPosition,
    hlsl::math::quaternion<hlsl::float64_t>& outOrientation,
    hlsl::float64_t* outAppliedDistance,
    hlsl::float64_t2* outOrbitUv)
{
    SCameraPathPose pathPose = {};
    if (!tryBuildPathPoseFromState(targetPosition, state, limits, frame, pathPose))
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
    return CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.s, rhs.s) <= thresholds.sToleranceDeg &&
        CCameraMathUtilities::nearlyEqualScalar(lhs.u, rhs.u, thresholds.scalarTolerance) &&
        CCameraMathUtilities::nearlyEqualScalar(lhs.v, rhs.v, thresholds.scalarTolerance) &&
        CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.roll, rhs.roll) <= thresholds.rollToleranceDeg;
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
    deltaVector.x = CCameraMathUtilities::wrapAngleRad(deltaVector.x);
    deltaVector.w = CCameraMathUtilities::wrapAngleRad(deltaVector.w);
    return deltaVector;
}

SCameraPathDelta CCameraPathUtilities::buildPathStateDelta(
    const ICamera::PathState& currentState,
    const ICamera::PathState& desiredState)
{
    return SCameraPathDelta::fromVector(buildPathStateDeltaVector(currentState, desiredState));
}

SCameraPathDelta CCameraPathUtilities::buildDefaultPathControlDelta(const SCameraPathControlContext& context)
{
    return context.requested;
}

bool CCameraPathUtilities::tryBuildCanonicalPathState(
    const hlsl::float64_t3& targetPosition,
    const ICamera::PathState& state,
    const SCameraPathLimits& limits,
    const SWorldFrame& frame,
    SCameraCanonicalPathState& outState)
{
    outState = {};
    if (!tryBuildPathPoseFromState(targetPosition, state, limits, frame, outState.pose))
        return false;

    outState.targetRelative = {
        .target = targetPosition,
        .angles = outState.pose.orbitUv,
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
    stateVector.x = CCameraMathUtilities::wrapAngleRad(stateVector.x);
    stateVector.w = CCameraMathUtilities::wrapAngleRad(stateVector.w);
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
        .s = CCameraMathUtilities::lerpWrappedAngleRad(fromVector.x, toVector.x, alpha),
        .u = fromVector.y + (toVector.y - fromVector.y) * alpha,
        .v = fromVector.z + (toVector.z - fromVector.z) * alpha,
        .roll = CCameraMathUtilities::lerpWrappedAngleRad(fromVector.w, toVector.w, alpha)
    };
}

bool CCameraPathUtilities::tryBuildPathStateTransition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& currentPosition,
    const hlsl::float64_t3& desiredPosition,
    const SCameraPathLimits& limits,
    const SWorldFrame& frame,
    const ICamera::PathState* currentStateOverride,
    const ICamera::PathState* desiredStateOverride,
    SCameraPathStateTransition& outTransition)
{
    if (!tryResolvePathState(targetPosition, currentPosition, limits, frame, currentStateOverride, outTransition.current))
        return false;
    if (!tryResolvePathState(targetPosition, desiredPosition, limits, frame, desiredStateOverride, outTransition.desired))
        return false;

    outTransition.delta = buildPathStateDelta(outTransition.current, outTransition.desired);
    return true;
}

SCameraPathModel CCameraPathUtilities::makeDefaultPathModel(const SWorldFrame& frame)
{
    return {
        .resolveState =
            [frame](const hlsl::float64_t3& targetPosition,
                const hlsl::float64_t3& position,
                const SCameraPathLimits& limits,
                const ICamera::PathState* requestedState,
                ICamera::PathState& outState) -> bool
            {
                return tryResolvePathState(targetPosition, position, limits, frame, requestedState, outState);
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
            [frame](const hlsl::float64_t3& targetPosition,
                const ICamera::PathState& state,
                const SCameraPathLimits& limits,
                SCameraCanonicalPathState& outState) -> bool
            {
                return tryBuildCanonicalPathState(targetPosition, state, limits, frame, outState);
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

} // namespace nbl::ext::cameras
