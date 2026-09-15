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
    static ICamera::PathState makeDefaultPathState(double minU = SCameraPathDefaults::MinU);

    /// @brief Build path-state comparison tolerances from caller-provided angular and scalar thresholds.
    static SCameraPathComparisonThresholds makePathComparisonThresholds(
        double angularToleranceDeg = SCameraPathDefaults::AngleToleranceDeg,
        double scalarTolerance = SCameraPathDefaults::ScalarTolerance);

    /// @brief Return the default path-state limits used when a camera does not expose custom ones.
    static inline constexpr SCameraPathLimits makeDefaultPathLimits()
    {
        return SCameraPathDefaults::Limits;
    }

    /// @brief Check whether every scalar stored in the path state is finite.
    static bool isPathStateFinite(const ICamera::PathState& state);

    /// @brief Check whether the path limits can be sanitized into a valid numeric domain.
    static bool isPathLimitsWellFormed(const SCameraPathLimits& limits);

    /// @brief Clamp and normalize path-state limits into a valid numeric domain.
    static bool sanitizePathLimits(SCameraPathLimits& limits);

    /// @brief Sanitize a path state against a caller-provided `minU` lower bound.
    static bool sanitizePathState(ICamera::PathState& state, double minU);

    /// @brief Sanitize a path state against a full limit bundle and optionally report the applied distance.
    static bool sanitizePathState(ICamera::PathState& state, const SCameraPathLimits& limits, double* outAppliedDistance = nullptr);

    /// @brief Rescale the `(u, v)` pair so the path state reaches the requested radial distance.
    static bool tryScalePathStateDistance(
        double desiredDistance,
        double minU,
        ICamera::PathState& ioState,
        double* outAppliedDistance = nullptr);

    /// @brief Update the distance encoded by a path state while respecting the provided limits.
    static bool tryUpdatePathStateDistance(
        float desiredDistance,
        const SCameraPathLimits& limits,
        ICamera::PathState& ioState,
        SCameraPathDistanceUpdateResult* outResult = nullptr);

    static bool tryBuildPathStateFromPosition(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        double minU,
        ICamera::PathState& outState);

    static bool tryResolvePathState(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const SCameraPathLimits& limits,
        const ICamera::PathState* requestedState,
        ICamera::PathState& outState);

    static bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraPathPose& outPose);

    static bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        hlsl::float64_t3& outPosition,
        hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation,
        hlsl::float64_t* outAppliedDistance = nullptr,
        hlsl::float64_t2* outOrbitUv = nullptr);

    static bool pathStatesNearlyEqual(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {});

    static bool pathStatesChanged(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {});

    static hlsl::float64_t4 buildPathStateDeltaVector(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState);

    static SCameraPathDelta buildPathStateDelta(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState);

    static SCameraPathDelta makePathDeltaFromVirtualPathMotion(
        const hlsl::float64_t3& translation,
        const hlsl::float64_t3& rotation = hlsl::float64_t3(0.0));

    static SCameraPathDelta buildDefaultPathControlDelta(const SCameraPathControlContext& context);

    static void appendPathDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        double moveDenominator,
        double rotationDenominator,
        const SCameraPathComparisonThresholds& thresholds = {});

    static bool tryBuildCanonicalPathState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraCanonicalPathState& outState);

    static bool tryApplyPathStateDelta(
        const ICamera::PathState& currentState,
        const SCameraPathDelta& delta,
        const SCameraPathLimits& limits,
        ICamera::PathState& outState);

    static ICamera::PathState blendPathStates(
        const ICamera::PathState& from,
        const ICamera::PathState& to,
        double alpha);

    static bool tryBuildPathStateTransition(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& currentPosition,
        const hlsl::float64_t3& desiredPosition,
        const SCameraPathLimits& limits,
        const ICamera::PathState* currentStateOverride,
        const ICamera::PathState* desiredStateOverride,
        SCameraPathStateTransition& outTransition);

    static SCameraPathModel makeDefaultPathModel();
};

} // namespace nbl::core

#endif // _C_CAMERA_PATH_UTILITIES_HPP_
