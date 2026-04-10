#ifndef _C_CAMERA_PATH_UTILITIES_HPP_
#define _C_CAMERA_PATH_UTILITIES_HPP_

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

#include "CCameraPathMetadata.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraVirtualEventUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Shared helpers for the reusable `PathRig` camera kind.
struct SCameraPathPose final : SCameraRigPose
{
    /// @brief Final radial distance actually applied after clamping and path-state sanitization.
    hlsl::float64_t appliedDistance = 0.0;
    /// @brief Canonical orbit yaw/pitch derived from the evaluated path state.
    hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
};

/// @brief Typed delta applied to `ICamera::PathState`.
struct SCameraPathDelta final : ICamera::PathState
{
    /// @brief Pack the delta into one four-component vector.
    inline hlsl::float64_t4 asVector() const
    {
        return ICamera::PathState::asVector();
    }

    /// @brief Reinterpret the delta as the translation-style helper representation.
    inline hlsl::float64_t3 translationVector() const
    {
        return ICamera::PathState::asTranslationVector();
    }

    /// @brief Rebuild the delta from the packed vector representation.
    static inline SCameraPathDelta fromVector(const hlsl::float64_t4& value)
    {
        SCameraPathDelta delta = {};
        delta.s = value.x;
        delta.u = value.y;
        delta.v = value.z;
        delta.roll = value.w;
        return delta;
    }

    /// @brief Rebuild the delta from a translation-style helper vector and optional roll value.
    static inline SCameraPathDelta fromMotion(const hlsl::float64_t3& translation, const double pathRoll = 0.0)
    {
        SCameraPathDelta delta = {};
        delta.s = translation.z;
        delta.u = translation.x;
        delta.v = translation.y;
        delta.roll = pathRoll;
        return delta;
    }
};

/// @brief One desired path-state change expressed as current state, desired state, and their delta.
struct SCameraPathStateTransition final
{
    ICamera::PathState current = {};
    ICamera::PathState desired = {};
    SCameraPathDelta delta = {};
};

/// @brief Canonical evaluated path state combining a final pose and target-relative view of that pose.
struct SCameraCanonicalPathState final
{
    SCameraPathPose pose = {};
    SCameraTargetRelativeState targetRelative = {};
};

/// @brief Comparison tolerances used when matching two path states.
struct SCameraPathComparisonThresholds final
{
    double sToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    double rollToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    double scalarTolerance = SCameraToolingThresholds::ScalarTolerance;
};

/// @brief Result of updating the path distance while preserving the rest of the path state.
struct SCameraPathDistanceUpdateResult final
{
    bool exact = false;
    hlsl::float64_t appliedDistance = 0.0;
};

/// @brief Default constants used by the built-in `Path Rig` model.
struct SCameraPathDefaults final
{
    static constexpr double MinU = static_cast<double>(SCameraTargetRelativeTraits::MinDistance);
    static constexpr double ScalarTolerance = SCameraToolingThresholds::ScalarTolerance;
    static constexpr double ExactStateTolerance = SCameraToolingThresholds::TinyScalarEpsilon;
    static constexpr double ExactAngleToleranceDeg = ExactStateTolerance * 180.0 / hlsl::numbers::pi<double>;
    static constexpr double AngleToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr std::string_view Identifier = SCameraPathRigMetadata::Identifier;
    static inline constexpr std::string_view Description = SCameraPathRigMetadata::DefaultModelDescription;
    static inline constexpr ICamera::PathStateLimits Limits = {};
    static inline constexpr SCameraPathComparisonThresholds ComparisonThresholds = {
        .sToleranceDeg = AngleToleranceDeg,
        .rollToleranceDeg = AngleToleranceDeg,
        .scalarTolerance = ScalarTolerance
    };
    static inline constexpr SCameraPathComparisonThresholds ExactComparisonThresholds = {
        .sToleranceDeg = ExactAngleToleranceDeg,
        .rollToleranceDeg = ExactAngleToleranceDeg,
        .scalarTolerance = ExactStateTolerance
    };
};

using SCameraPathLimits = ICamera::PathStateLimits;

/// @brief Evaluation context passed into the active path-model control law.
struct SCameraPathControlContext final
{
    ICamera::PathState currentState = {};
    hlsl::float64_t3 translation = hlsl::float64_t3(0.0);
    hlsl::float64_t3 rotation = hlsl::float64_t3(0.0);
    hlsl::float64_t3 targetPosition = hlsl::float64_t3(0.0);
    const CReferenceTransform* reference = nullptr;
    SCameraPathLimits limits = SCameraPathDefaults::Limits;
};

/// @brief Callback bundle defining path-state resolution, input response, evaluation, and distance updates.
///
/// A concrete `Path Rig` model provides:
/// - state resolution from target position, world position, and optional typed input
/// - one control law turning accumulated runtime motion into `SCameraPathDelta`
/// - one state integrator
/// - one canonical evaluator producing pose and target-relative view data
/// - one distance-update rule for typed helpers that adjust distance directly
struct SCameraPathModel final
{
    using resolve_state_t = std::function<bool(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const SCameraPathLimits& limits,
        const ICamera::PathState* requestedState,
        ICamera::PathState& outState)>;
    using control_law_t = std::function<SCameraPathDelta(const SCameraPathControlContext&)>;
    using integrate_t = std::function<bool(
        const ICamera::PathState& currentState,
        const SCameraPathDelta& delta,
        const SCameraPathLimits& limits,
        ICamera::PathState& outState)>;
    using evaluate_t = std::function<bool(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraCanonicalPathState& outState)>;
    using update_distance_t = std::function<bool(
        const float desiredDistance,
        const SCameraPathLimits& limits,
        ICamera::PathState& ioState,
        SCameraPathDistanceUpdateResult* outResult)>;

    resolve_state_t resolveState;
    control_law_t controlLaw;
    integrate_t integrate;
    evaluate_t evaluate;
    update_distance_t updateDistance;
};

/// @brief Shared state, comparison, and model-building helpers for `Path Rig`.
struct CCameraPathUtilities final
{
    /// @brief Build the default path state used by the built-in model.
    static inline ICamera::PathState makeDefaultPathState(const double minU = SCameraPathDefaults::MinU)
    {
        return {
            .s = 0.0,
            .u = minU,
            .v = 0.0,
            .roll = 0.0
        };
    }

    /// @brief Build path-state comparison tolerances from caller-provided angular and scalar thresholds.
    static inline SCameraPathComparisonThresholds makePathComparisonThresholds(
        const double angularToleranceDeg = SCameraPathDefaults::AngleToleranceDeg,
        const double scalarTolerance = SCameraPathDefaults::ScalarTolerance)
    {
        return {
            .sToleranceDeg = angularToleranceDeg,
            .rollToleranceDeg = angularToleranceDeg,
            .scalarTolerance = scalarTolerance
        };
    }

    /// @brief Return the default path-state limits used when a camera does not expose custom ones.
    static inline constexpr SCameraPathLimits makeDefaultPathLimits()
    {
        return SCameraPathDefaults::Limits;
    }

    /// @brief Check whether every scalar stored in the path state is finite.
    static inline bool isPathStateFinite(const ICamera::PathState& state)
    {
        return hlsl::CCameraMathUtilities::isFiniteScalar(state.s) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.u) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.v) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.roll);
    }

    /// @brief Check whether the path limits can be sanitized into a valid numeric domain.
    static inline bool isPathLimitsWellFormed(const SCameraPathLimits& limits)
    {
        return hlsl::CCameraMathUtilities::isFiniteScalar(limits.minU) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(limits.minDistance) &&
            !std::isnan(static_cast<double>(limits.maxDistance));
    }

    /// @brief Clamp and normalize path-state limits into a valid numeric domain.
    static inline bool sanitizePathLimits(SCameraPathLimits& limits)
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

    /// @brief Sanitize a path state against a caller-provided `minU` lower bound.
    static inline bool sanitizePathState(ICamera::PathState& state, const double minU)
    {
        return hlsl::CCameraMathUtilities::sanitizePathState(state.s, state.u, state.v, state.roll, minU);
    }

    /// @brief Sanitize a path state against a full limit bundle and optionally report the applied distance.
    static inline bool sanitizePathState(ICamera::PathState& state, const SCameraPathLimits& limits, double* outAppliedDistance = nullptr)
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

    /// @brief Rescale the `(u, v)` pair so the path state reaches the requested radial distance.
    static inline bool tryScalePathStateDistance(
        const double desiredDistance,
        const double minU,
        ICamera::PathState& ioState,
        double* outAppliedDistance = nullptr)
    {
        return hlsl::CCameraMathUtilities::tryScalePathStateDistance(
            desiredDistance,
            minU,
            ioState.u,
            ioState.v,
            outAppliedDistance);
    }

    /// @brief Update the distance encoded by a path state while respecting the provided limits.
    static inline bool tryUpdatePathStateDistance(
        const float desiredDistance,
        const SCameraPathLimits& limits,
        ICamera::PathState& ioState,
        SCameraPathDistanceUpdateResult* outResult = nullptr)
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

    static inline bool tryBuildPathStateFromPosition(
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

    static inline bool tryResolvePathState(
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

    static inline bool tryBuildPathPoseFromState(
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

    static inline bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        hlsl::float64_t3& outPosition,
        hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation,
        hlsl::float64_t* outAppliedDistance = nullptr,
        hlsl::float64_t2* outOrbitUv = nullptr)
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

    static inline bool pathStatesNearlyEqual(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        return hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.s, rhs.s) <= thresholds.sToleranceDeg &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.u, rhs.u, thresholds.scalarTolerance) &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.v, rhs.v, thresholds.scalarTolerance) &&
            hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.roll, rhs.roll) <= thresholds.rollToleranceDeg;
    }

    static inline bool pathStatesChanged(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        return !pathStatesNearlyEqual(lhs, rhs, thresholds);
    }

    static inline hlsl::float64_t4 buildPathStateDeltaVector(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        auto deltaVector = desiredState.asVector() - currentState.asVector();
        deltaVector.x = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.x);
        deltaVector.w = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.w);
        return deltaVector;
    }

    static inline SCameraPathDelta buildPathStateDelta(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        return SCameraPathDelta::fromVector(buildPathStateDeltaVector(currentState, desiredState));
    }

    static inline SCameraPathDelta makePathDeltaFromVirtualPathMotion(
        const hlsl::float64_t3& translation,
        const hlsl::float64_t3& rotation = hlsl::float64_t3(0.0))
    {
        return SCameraPathDelta::fromMotion(translation, rotation.z);
    }

    static inline SCameraPathDelta buildDefaultPathControlDelta(const SCameraPathControlContext& context)
    {
        return makePathDeltaFromVirtualPathMotion(context.translation, context.rotation);
    }

    static inline void appendPathDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        const double moveDenominator,
        const double rotationDenominator,
        const SCameraPathComparisonThresholds& thresholds = {})
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

    static inline bool tryBuildCanonicalPathState(
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

    static inline bool tryApplyPathStateDelta(
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

    static inline ICamera::PathState blendPathStates(
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

    static inline bool tryBuildPathStateTransition(
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

    static inline SCameraPathModel makeDefaultPathModel()
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
};

} // namespace nbl::core

#endif // _C_CAMERA_PATH_UTILITIES_HPP_
