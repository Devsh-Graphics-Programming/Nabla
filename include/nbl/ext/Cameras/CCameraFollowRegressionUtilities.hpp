// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

#include <string>

#include "CCameraFollowUtilities.hpp"

namespace nbl::ext::cameras
{

struct SCameraProjectedTargetMetrics final
{
    hlsl::float32_t2 ndc = hlsl::float32_t2(0.0f);
    float radius = 0.0f;
};

/// @brief Reusable follow validation helpers.
///
/// The checks stay camera-domain:
///
/// - camera-to-target direction must match the camera forward axis for locking modes
/// - target distance must be finite and internally consistent
/// - spherical cameras must write the tracked target back into spherical target state
/// - spherical distance must match the goal-derived distance when present
struct SCameraFollowRegressionResult
{
    bool passed = false;
    bool hasLockMetrics = false;
    hlsl::float64_t lockAngleDeg = 0.0;
    hlsl::float64_t targetDistance = 0.0;
    bool hasProjectedMetrics = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
    bool hasSphericalState = false;
    hlsl::float64_t3 sphericalTarget = hlsl::float64_t3(0.0);
    hlsl::float64_t sphericalDistance = 0.0;
};

/// @brief Reusable visual/debug metrics for one active follow configuration.
struct SCameraFollowVisualMetrics
{
    bool active = false;
    ECameraFollowMode mode = ECameraFollowMode::Unknown;
    bool lockValid = false;
    hlsl::float64_t lockAngleDeg = 0.0;
    hlsl::float64_t targetDistance = 0.0;
    bool projectedValid = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
};

/// @brief Shared view/projection bundle for CPU-side projected target metrics.
struct SCameraProjectionContext
{
    hlsl::float32_t4x4 viewMatrix = hlsl::float32_t4x4(1.0f);
    hlsl::float32_t4x4 projectionMatrix = hlsl::float32_t4x4(1.0f);
};

/// @brief Shared tolerances for follow target lock, writeback, and projected-center checks.
struct SCameraFollowRegressionThresholds
{
    static inline constexpr float DefaultClipWEpsilon = 1e-5f;
    static inline constexpr float DefaultProjectedNdcTolerance = 0.03f;
    static inline constexpr hlsl::float64_t DefaultLockAngleToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultDistanceTolerance = SCameraToolingThresholds::ScalarTolerance;
    static inline constexpr double DefaultTargetTolerance = SCameraToolingThresholds::TinyScalarEpsilon;
    static inline constexpr double DefaultPositionTolerance = SCameraToolingThresholds::DefaultPositionTolerance;
    static inline constexpr double DefaultRotationToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultScalarTolerance = SCameraToolingThresholds::ScalarTolerance;

    float clipWEpsilon = DefaultClipWEpsilon;
    float projectedNdcTolerance = DefaultProjectedNdcTolerance;
    hlsl::float64_t lockAngleToleranceDeg = DefaultLockAngleToleranceDeg;
    double distanceTolerance = DefaultDistanceTolerance;
    double targetTolerance = DefaultTargetTolerance;
    double positionTolerance = DefaultPositionTolerance;
    double rotationToleranceDeg = DefaultRotationToleranceDeg;
    double scalarTolerance = DefaultScalarTolerance;
};

/// @brief Bundled reusable follow regression flow.
/// The helper builds a follow goal, applies it, verifies the resulting camera state,
/// and then checks lock/writeback follow consistency.
struct SCameraFollowApplyValidationResult
{
    bool hasGoal = false;
    CCameraGoal goal = {};
    CCameraGoalSolver::SApplyResult applyResult = {};
    bool hasCapturedGoal = false;
    CCameraGoal capturedGoal = {};
    SCameraFollowRegressionResult regression = {};
};

struct CCameraFollowRegressionUtilities final
{
public:
    static SCameraFollowRegressionThresholds makeFollowRegressionThresholds(
        float projectedNdcTolerance = SCameraFollowRegressionThresholds::DefaultProjectedNdcTolerance,
        hlsl::float64_t lockAngleToleranceDeg = SCameraFollowRegressionThresholds::DefaultLockAngleToleranceDeg);

    static bool tryComputeProjectedFollowTargetMetrics(
        const SCameraProjectionContext& projectionContext,
        const CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        float clipWEpsilon = SCameraFollowRegressionThresholds::DefaultClipWEpsilon);

    /// @brief Check that the tracked target projects close enough to the screen centre.
    /// @param error optional (may be null); receives a description of the first failure.
    static bool validateProjectedFollowTargetContract(
        const SCameraProjectionContext& projectionContext,
        const CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        std::string* error = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static SCameraFollowVisualMetrics buildFollowVisualMetrics(
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig* followConfig,
        const SCameraProjectionContext* projectionContext = nullptr);

    static bool validateFollowTargetContract(
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& followConfig,
        const CCameraGoal& followGoal,
        SCameraFollowRegressionResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static bool buildApplyAndValidateFollowTargetContract(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& followConfig,
        SCameraFollowApplyValidationResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

