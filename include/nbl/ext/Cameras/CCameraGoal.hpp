// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_GOAL_HPP_
#define _C_CAMERA_GOAL_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "CCameraPathUtilities.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Typed transport object for camera state used by capture, comparison, presets, and playback.
struct CCameraGoal : SCameraRigPose
{
    /// @brief Camera kind that originally produced this goal.
    ICamera::CameraKind sourceKind = ICamera::CameraKind::Unknown;
    /// @brief Capability mask captured from the source camera.
    ICamera::capability_flags_t sourceCapabilities = ICamera::None;
    /// @brief Goal-state fragments that were valid on the source camera.
    ICamera::goal_state_flags_t sourceGoalStateMask = ICamera::GoalStateNone;
    /// @brief Whether `targetPosition` is present in this goal.
    bool hasTargetPosition = false;
    /// @brief Tracked target position in world space.
    hlsl::float64_t3 targetPosition = hlsl::float64_t3(0.0);
    /// @brief Whether `distance` is present in this goal.
    bool hasDistance = false;
    /// @brief Explicit target-relative distance when present.
    float distance = 0.f;
    /// @brief Whether the canonical orbit state is present in this goal.
    bool hasOrbitState = false;
    /// @brief Canonical orbit yaw and pitch, expressed in radians.
    hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
    /// @brief Distance associated with `orbitUv` when the orbit state is present.
    float orbitDistance = 0.f;
    /// @brief Whether a typed path state is present in this goal.
    bool hasPathState = false;
    /// @brief Typed path state captured from or authored for a `Path Rig` camera.
    ICamera::PathState pathState = {};
    /// @brief Whether a dynamic perspective state is present in this goal.
    bool hasDynamicPerspectiveState = false;
    /// @brief Typed dynamic perspective state captured from or authored for the source camera.
    ICamera::DynamicPerspectiveState dynamicPerspectiveState = {};
};

/// @brief Shared canonicalization, comparison, and conversion helpers for `CCameraGoal`.
struct CCameraGoalUtilities final
{
public:
    /// @brief Compute which typed goal-state fragments are required by the current goal payload.
    static inline ICamera::goal_state_flags_t getRequiredGoalStateMask(const CCameraGoal& target)
    {
        ICamera::goal_state_flags_t mask = ICamera::GoalStateNone;
        if (target.hasTargetPosition || target.hasDistance || target.hasOrbitState)
            mask |= ICamera::GoalStateSphericalTarget;
        if (target.hasDynamicPerspectiveState)
            mask |= ICamera::GoalStateDynamicPerspective;
        if (target.hasPathState)
            mask |= ICamera::GoalStatePath;
        return mask;
    }

    /// @brief Overwrite the canonical target-relative fields of a goal from prebuilt state and pose data.
    static inline void applyCanonicalTargetRelativeGoalFields(
        CCameraGoal& goal,
        const SCameraTargetRelativeState& state,
        const SCameraTargetRelativePose& pose)
    {
        goal.position = pose.position;
        goal.orientation = pose.orientation;
        goal.hasTargetPosition = true;
        goal.targetPosition = state.target;
        goal.hasDistance = true;
        goal.distance = static_cast<float>(pose.appliedDistance);
        goal.hasOrbitState = true;
        goal.orbitUv = state.orbitUv;
        goal.orbitDistance = static_cast<float>(pose.appliedDistance);
    }

    /// @brief Rebuild the canonical target-relative portion of a goal from typed target-relative state.
    static inline bool applyCanonicalTargetRelativeGoal(CCameraGoal& goal, const SCameraTargetRelativeState& state)
    {
        SCameraTargetRelativePose pose = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativePoseFromState(state, SCameraTargetRelativeTraits::MinDistance, SCameraTargetRelativeTraits::DefaultMaxDistance, pose))
            return false;

        applyCanonicalTargetRelativeGoalFields(goal, state, pose);
        return true;
    }

    /// @brief Rebuild the canonical pose and orbit fields of a goal from typed path state.
    static inline bool applyCanonicalPathGoalFields(
        CCameraGoal& goal,
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& pathState,
        const SCameraPathLimits& limits = CCameraPathUtilities::makeDefaultPathLimits())
    {
        SCameraCanonicalPathState canonicalPathState = {};
        if (!CCameraPathUtilities::tryBuildCanonicalPathState(targetPosition, pathState, limits, canonicalPathState))
            return false;

        goal.hasTargetPosition = true;
        goal.targetPosition = targetPosition;
        goal.hasPathState = true;
        goal.pathState = pathState;
        SCameraTargetRelativePose canonicalPose = {};
        canonicalPose.position = canonicalPathState.pose.position;
        canonicalPose.orientation = canonicalPathState.pose.orientation;
        canonicalPose.appliedDistance = canonicalPathState.pose.appliedDistance;
        applyCanonicalTargetRelativeGoalFields(
            goal,
            canonicalPathState.targetRelative,
            canonicalPose);
        return true;
    }

    /// @brief Rebuild the canonical pose fields from the goal's current spherical-target payload.
    static inline bool applyCanonicalSphericalGoal(CCameraGoal& goal)
    {
        if (!(goal.hasTargetPosition && goal.hasOrbitState))
            return false;
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitUv.x) || !hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitUv.y) || !hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitDistance))
            return false;

        return applyCanonicalTargetRelativeGoal(
            goal,
            {
                .target = goal.targetPosition,
                .orbitUv = goal.orbitUv,
                .distance = goal.orbitDistance
            });
    }

    /// @brief Infer a target-relative goal from a target position and a desired camera position.
    static inline bool buildCanonicalTargetRelativeGoalFromPosition(
        CCameraGoal& goal,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position)
    {
        SCameraTargetRelativeState state = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeStateFromPosition(
                targetPosition,
                position,
                SCameraTargetRelativeTraits::MinDistance,
                SCameraTargetRelativeTraits::DefaultMaxDistance,
                state))
        {
            return false;
        }

        return applyCanonicalTargetRelativeGoal(goal, state);
    }

    /// @brief Resolve the effective target-relative state of a goal against the current camera state.
    static inline bool tryResolveCanonicalTargetRelativeState(
        const CCameraGoal& goal,
        const ICamera::SphericalTargetState& currentState,
        SCameraTargetRelativeState& outState)
    {
        outState.target = goal.hasTargetPosition ? goal.targetPosition : currentState.target;
        outState.orbitUv = currentState.orbitUv;
        outState.distance = currentState.distance;

        if (goal.hasOrbitState)
        {
            outState.orbitUv = goal.orbitUv;
            outState.distance = goal.orbitDistance;
        }
        else
        {
            SCameraTargetRelativeState resolvedState = {};
            if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeStateFromPosition(
                    outState.target,
                    goal.position,
                    currentState.minDistance,
                    currentState.maxDistance,
                    resolvedState))
            {
                return false;
            }

            outState.orbitUv = resolvedState.orbitUv;
            outState.distance = resolvedState.distance;
        }

        if (goal.hasDistance && !goal.hasOrbitState)
            outState.distance = goal.distance;

        outState.distance = std::clamp(outState.distance, currentState.minDistance, currentState.maxDistance);
        return true;
    }

    /// @brief Rebuild the canonical pose fields from the goal's current path payload.
    static inline bool applyCanonicalPathGoal(CCameraGoal& goal)
    {
        if (!(goal.hasPathState && goal.hasTargetPosition))
            return false;
        if (!CCameraPathUtilities::isPathStateFinite(goal.pathState))
            return false;
        return applyCanonicalPathGoalFields(goal, goal.targetPosition, goal.pathState);
    }

    /// @brief Canonicalize whichever typed state fragments are currently present on the goal.
    static inline bool applyCanonicalGoalState(CCameraGoal& goal)
    {
        if (goal.hasPathState)
            return applyCanonicalPathGoal(goal);

        if (goal.hasTargetPosition && goal.hasOrbitState)
            return applyCanonicalSphericalGoal(goal);

        return true;
    }

    /// @brief Return a value-copied goal after canonicalizing its typed state.
    static inline CCameraGoal canonicalizeGoal(CCameraGoal goal)
    {
        applyCanonicalGoalState(goal);
        return goal;
    }

    /// @brief Check whether every populated scalar and vector stored by the goal is finite.
    static inline bool isGoalFinite(const CCameraGoal& goal)
    {
        if (!hlsl::CCameraMathUtilities::isFiniteVec3(goal.position) || !hlsl::CCameraMathUtilities::isFiniteQuaternion(goal.orientation))
            return false;
        if (goal.hasTargetPosition && !hlsl::CCameraMathUtilities::isFiniteVec3(goal.targetPosition))
            return false;
        if (goal.hasDistance && !hlsl::CCameraMathUtilities::isFiniteScalar(goal.distance))
            return false;
        if (goal.hasOrbitState && (!hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitUv.x) || !hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitUv.y) || !hlsl::CCameraMathUtilities::isFiniteScalar(goal.orbitDistance)))
            return false;
        if (goal.hasPathState && !CCameraPathUtilities::isPathStateFinite(goal.pathState))
            return false;
        if (goal.hasDynamicPerspectiveState &&
            (!hlsl::CCameraMathUtilities::isFiniteScalar(goal.dynamicPerspectiveState.baseFov) || !hlsl::CCameraMathUtilities::isFiniteScalar(goal.dynamicPerspectiveState.referenceDistance)))
            return false;
        return true;
    }

    /// @brief Compare two goals using caller-provided pose and scalar tolerances.
    static inline bool compareGoals(const CCameraGoal& actual, const CCameraGoal& expected,
        const double posEps, const double rotEpsDeg, const double scalarEps)
    {
        hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
        if (!hlsl::CCameraMathUtilities::tryComputePoseDelta(actual.position, actual.orientation, expected.position, expected.orientation, poseDelta))
            return false;
        if (poseDelta.position > posEps || poseDelta.rotationDeg > rotEpsDeg)
            return false;

        if (expected.hasTargetPosition)
        {
            if (!actual.hasTargetPosition || !hlsl::CCameraMathUtilities::nearlyEqualVec3(actual.targetPosition, expected.targetPosition, scalarEps))
                return false;
        }
        if (expected.hasDistance)
        {
            if (!actual.hasDistance || !hlsl::CCameraMathUtilities::nearlyEqualScalar(static_cast<double>(actual.distance), static_cast<double>(expected.distance), scalarEps))
                return false;
        }
        if (expected.hasOrbitState)
        {
            if (!actual.hasOrbitState)
                return false;
            if (hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(expected.orbitUv.x, actual.orbitUv.x) > rotEpsDeg)
                return false;
            if (hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(expected.orbitUv.y, actual.orbitUv.y) > rotEpsDeg)
                return false;
            if (!hlsl::CCameraMathUtilities::nearlyEqualScalar(static_cast<double>(actual.orbitDistance), static_cast<double>(expected.orbitDistance), scalarEps))
                return false;
        }
        if (expected.hasPathState)
        {
            if (!actual.hasPathState)
                return false;
            if (!CCameraPathUtilities::pathStatesNearlyEqual(actual.pathState, expected.pathState, CCameraPathUtilities::makePathComparisonThresholds(rotEpsDeg, scalarEps)))
                return false;
        }
        if (expected.hasDynamicPerspectiveState)
        {
            if (!actual.hasDynamicPerspectiveState)
                return false;
            if (!hlsl::CCameraMathUtilities::nearlyEqualScalar(static_cast<double>(actual.dynamicPerspectiveState.baseFov), static_cast<double>(expected.dynamicPerspectiveState.baseFov), scalarEps))
                return false;
            if (!hlsl::CCameraMathUtilities::nearlyEqualScalar(static_cast<double>(actual.dynamicPerspectiveState.referenceDistance), static_cast<double>(expected.dynamicPerspectiveState.referenceDistance), scalarEps))
                return false;
        }

        return true;
    }

    static inline std::string describeGoalMismatch(const CCameraGoal& actual, const CCameraGoal& expected)
    {
        std::ostringstream oss;
        hlsl::SCameraPoseDelta<hlsl::float64_t> poseDelta = {};
        const bool hasPoseDelta = hlsl::CCameraMathUtilities::tryComputePoseDelta(actual.position, actual.orientation, expected.position, expected.orientation, poseDelta);
        const auto currentOrientation = hlsl::CCameraMathUtilities::normalizeQuaternion(actual.orientation);
        const auto expectedOrientation = hlsl::CCameraMathUtilities::normalizeQuaternion(expected.orientation);
        oss << "pos_delta=" << (hasPoseDelta ? poseDelta.position : std::numeric_limits<double>::quiet_NaN())
            << " rot_delta_deg=" << (hasPoseDelta ? poseDelta.rotationDeg : std::numeric_limits<double>::quiet_NaN())
            << " current_pos=(" << actual.position.x << "," << actual.position.y << "," << actual.position.z << ")"
            << " expected_pos=(" << expected.position.x << "," << expected.position.y << "," << expected.position.z << ")"
            << " current_quat=(" << currentOrientation.data.x << "," << currentOrientation.data.y << "," << currentOrientation.data.z << "," << currentOrientation.data.w << ")"
            << " expected_quat=(" << expectedOrientation.data.x << "," << expectedOrientation.data.y << "," << expectedOrientation.data.z << "," << expectedOrientation.data.w << ")";

        if (actual.hasTargetPosition)
        {
            oss << " target=(" << actual.targetPosition.x << "," << actual.targetPosition.y << "," << actual.targetPosition.z << ")";
            if (actual.hasDistance)
                oss << " distance=" << actual.distance;
            if (actual.hasOrbitState)
                oss << " orbit_u=" << actual.orbitUv.x << " orbit_v=" << actual.orbitUv.y;
        }
        else if (expected.hasTargetPosition || expected.hasDistance || expected.hasOrbitState)
        {
            oss << " spherical_state=unavailable";
        }
        if (actual.hasPathState)
        {
            oss << " path_s=" << actual.pathState.s
                << " path_u=" << actual.pathState.u
                << " path_v=" << actual.pathState.v
                << " path_roll=" << actual.pathState.roll;
        }
        else if (expected.hasPathState)
        {
            oss << " path_state=unavailable";
        }

        if (actual.hasDynamicPerspectiveState)
        {
            oss << " dynamic_base_fov=" << actual.dynamicPerspectiveState.baseFov
                << " dynamic_reference_distance=" << actual.dynamicPerspectiveState.referenceDistance;
        }
        else if (expected.hasDynamicPerspectiveState)
        {
            oss << " dynamic_perspective_state=unavailable";
        }

        return oss.str();
    }

    static inline CCameraGoal blendGoals(const CCameraGoal& a, const CCameraGoal& b, double alpha)
    {
        CCameraGoal blended;
        blended.position = a.position + (b.position - a.position) * alpha;
        blended.orientation = hlsl::CCameraMathUtilities::slerpQuaternion(a.orientation, b.orientation, static_cast<hlsl::float64_t>(alpha));
        blended.sourceKind = (a.sourceKind == b.sourceKind) ? a.sourceKind : ICamera::CameraKind::Unknown;
        blended.sourceCapabilities = a.sourceCapabilities & b.sourceCapabilities;
        blended.sourceGoalStateMask = a.sourceGoalStateMask | b.sourceGoalStateMask;
        blended.hasTargetPosition = a.hasTargetPosition || b.hasTargetPosition;
        if (blended.hasTargetPosition)
        {
            const auto ta = a.hasTargetPosition ? a.targetPosition : b.targetPosition;
            const auto tb = b.hasTargetPosition ? b.targetPosition : a.targetPosition;
            blended.targetPosition = ta + (tb - ta) * alpha;
        }
        blended.hasDistance = a.hasDistance || b.hasDistance;
        if (blended.hasDistance)
        {
            const float da = a.hasDistance ? a.distance : b.distance;
            const float db = b.hasDistance ? b.distance : a.distance;
            blended.distance = da + (db - da) * static_cast<float>(alpha);
        }
        blended.hasOrbitState = a.hasOrbitState || b.hasOrbitState;
        if (blended.hasOrbitState)
        {
            const auto orbitUvA = a.hasOrbitState ? a.orbitUv : b.orbitUv;
            const auto orbitUvB = b.hasOrbitState ? b.orbitUv : a.orbitUv;
            const float da = a.hasOrbitState ? a.orbitDistance : b.orbitDistance;
            const float db = b.hasOrbitState ? b.orbitDistance : a.orbitDistance;

            blended.orbitUv = hlsl::float64_t2(
                hlsl::CCameraMathUtilities::lerpWrappedAngleRad(orbitUvA.x, orbitUvB.x, alpha),
                hlsl::CCameraMathUtilities::lerpWrappedAngleRad(orbitUvA.y, orbitUvB.y, alpha));
            blended.orbitDistance = da + (db - da) * static_cast<float>(alpha);
        }
        blended.hasDynamicPerspectiveState = a.hasDynamicPerspectiveState || b.hasDynamicPerspectiveState;
        if (blended.hasDynamicPerspectiveState)
        {
            const auto dynamicA = a.hasDynamicPerspectiveState ? a.dynamicPerspectiveState : b.dynamicPerspectiveState;
            const auto dynamicB = b.hasDynamicPerspectiveState ? b.dynamicPerspectiveState : a.dynamicPerspectiveState;
            blended.dynamicPerspectiveState.baseFov = dynamicA.baseFov + (dynamicB.baseFov - dynamicA.baseFov) * static_cast<float>(alpha);
            blended.dynamicPerspectiveState.referenceDistance =
                dynamicA.referenceDistance + (dynamicB.referenceDistance - dynamicA.referenceDistance) * static_cast<float>(alpha);
        }
        blended.hasPathState = a.hasPathState || b.hasPathState;
        if (blended.hasPathState)
        {
            const auto pathA = a.hasPathState ? a.pathState : b.pathState;
            const auto pathB = b.hasPathState ? b.pathState : a.pathState;
            blended.pathState = CCameraPathUtilities::blendPathStates(pathA, pathB, alpha);
        }
        return canonicalizeGoal(blended);
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_HPP_

